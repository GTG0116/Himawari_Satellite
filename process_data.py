import boto3
import gc
import io
import shutil
import json
import urllib.request
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
import numpy as np
import os
from datetime import datetime, timezone, timedelta
from botocore import UNSIGNED
from botocore.config import Config
from PIL import Image
from scipy.ndimage import uniform_filter, zoom
from satpy import Scene

OUTPUT_DIR = 'site/data'
os.makedirs(OUTPUT_DIR, exist_ok=True)

BUCKET = 'noaa-himawari9'
MAX_FRAMES = 10  # rolling frame buffer per product

SAT_LON = 140.7  # Himawari-9 sub-satellite longitude (°E)

# Geographic extent: [west_lon, east_lon, south_lat, north_lat]
# This MUST match the imageBounds in site/index.html
# Himawari-9 full disk: 80°E to 200°E (=160°W), 60°S to 60°N
EXTENT = [80, 200, -60, 60]

# Maximum pixels per dimension after loading a band.
# Our output figure is 12 in × 150 DPI = 1 800 px wide, so 2 750 px input
# is already more than enough and keeps each band under ~30 MB of RAM.
# Band 3 (0.5 km, 11 000×11 000) is downsampled 4× to ~2 750×2 750;
# 2 km bands (5 500×5 500) are halved to ~2 750×2 750.
MAX_BAND_PX = 2750

# ---------------------------------------------------------------------------
# Cyclone sector settings
# ---------------------------------------------------------------------------
# URL for JTWC active storm data
STORMS_JSON_URL = ('https://raw.githubusercontent.com/GTG0116/JTWCTyphoonData/'
                   'claude/jtwc-forecast-viewer-NQqQA/data/storms.json')

# Padding (degrees) added around a storm's position to define the sector.
# 8° gives a ~16°×16° box, comfortably containing most wind-radii extents.
SECTOR_PAD_DEG = 8.0

# A sector clipped by the domain edge is only worth rendering if this many
# degrees survive in both axes; anything thinner is a sliver of limb.
MIN_SECTOR_DEG = 2.0

# Sector rendering uses a smaller figure at higher DPI to maximise pixel
# density over the small geographic area.
SECTOR_FIG_WIDTH = 8.0   # inches
SECTOR_DPI       = 200   # → 1 600 px wide for ~16° lon ≈ 100 px/° vs 15 px/° full disk


# ---------------------------------------------------------------------------
# Experimental precipitation-rate estimator configuration
# ---------------------------------------------------------------------------
# Look this far back for a previous Band 13 scan to derive a cloud-top cooling
# rate.  Himawari-9 full-disk cadence is 10 minutes, so 30 minutes ≈ 3 scans —
# long enough for a measurable temperature trend, short enough that the cloud
# field hasn't advected far.  Set to 0 to disable the cooling-rate term.
PRECIP_COOLING_MINUTES = 30

# Rain rates are reported in inches per hour.  The underlying regression is
# published in mm/hr, so it is converted on the way out.
MM_PER_INCH = 25.4

# Rain rates below this (in/hr) are treated as no rain and drawn transparent.
# 0.01 in/hr is the conventional "trace" threshold.
PRECIP_MIN_RATE = 0.01
# Hard ceiling (in/hr ≈ 102 mm/hr).  The bare regression curve runs away below
# ~195 K (hundreds of mm/hr), which is physically implausible; the operational
# Hydro-Estimator applies a similar cap.
PRECIP_MAX_RATE = 4.0
# Cloud-top temperatures warmer than this are assumed non-precipitating.
PRECIP_CLOUD_MAX_K = 260.0
# Diameter (km) of the neighbourhood used for the cloud-top texture test.
PRECIP_TEXTURE_KM = 100.0

# Himawari-9 geostationary geometry, used for the view-angle taper.  These are
# the values AHI HSD files carry in their projection metadata.
SAT_HEIGHT   = 35785863.0  # perspective point height, m
EARTH_RADIUS = 6378137.0   # GRS80 semi-major axis, m


# Module-level cache so each band is downloaded and loaded only once per run,
# even if it is needed by multiple products (e.g. Band 13 for both the IR
# product and the GeoColor cloud-top enhancement).
_band_cache: dict = {}


# ---------------------------------------------------------------------------
# Custom colour maps
# ---------------------------------------------------------------------------

def _ir_colormap():
    """NWS-style rainbow IR enhancement.

    Maps brightness temperature (190 K → 310 K):
      Cold cloud tops  (190–220 K) → white / magenta / red / orange
      Moderate clouds  (230–260 K) → orange / green / cyan
      Warm clear sky   (270–310 K) → blue → dark blue → near-black
    """
    return LinearSegmentedColormap.from_list('ir_enhancement', [
        (0.00, '#ffffff'),  # 190 K  – white  (extreme cold tops)
        (0.07, '#dd00dd'),  # 199 K  – magenta
        (0.17, '#ff0000'),  # 210 K  – red
        (0.27, '#ff5500'),  # 222 K  – orange-red
        (0.37, '#ff8800'),  # 233 K  – orange (was amber, removed yellow cast)
        (0.45, '#44cc00'),  # 244 K  – green  (was yellow #ffff00 – removed)
        (0.53, '#00cc00'),  # 254 K  – green
        (0.62, '#00cccc'),  # 264 K  – cyan
        (0.72, '#0066ff'),  # 276 K  – blue
        (0.87, '#001177'),  # 294 K  – dark blue
        (1.00, '#060606'),  # 310 K  – near-black
    ])


def _wv_colormap():
    """Water-vapour enhancement colormap.

    Maps brightness temperature (195 K → 280 K):
      Cold / moist upper troposphere (195–225 K) → deep navy → royal blue
      Moderate moisture              (225–250 K) → medium blue → teal
      Warm / dry troposphere         (250–280 K) → green → orange → red
    """
    return LinearSegmentedColormap.from_list('wv_enhancement', [
        (0.00, '#00003c'),  # 195 K  – deep navy
        (0.18, '#0000cc'),  # 209 K  – royal blue
        (0.35, '#0066ee'),  # 222 K  – medium blue
        (0.50, '#00bbdd'),  # 233 K  – light blue / cyan
        (0.63, '#00bb66'),  # 242 K  – teal-green
        (0.74, '#22cc00'),  # 250 K  – green (was yellow-green #aadd00 – removed)
        (0.84, '#ff8800'),  # 258 K  – orange (was yellow #ffcc00 – removed)
        (0.92, '#ff5500'),  # 265 K  – deep orange
        (1.00, '#cc1100'),  # 280 K  – red-orange (warm / dry)
    ])


# ---------------------------------------------------------------------------
# Precipitation rate scale
# ---------------------------------------------------------------------------
# Discrete radar-style rain-rate bins (in/hr).  N+1 edges → N colour bins.
# Edges 0, 2, 4, 6, 8 and 10 land on the legend ticks in site/index.html.
PRECIP_LEVELS = [0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 0.75, 1.25, 2.00, 3.00, 4.00]

# One colour per bin, ramping cyan → blue → green → yellow → red → magenta.
# Unlike the IR/WV enhancements, yellow is kept here: on a rain-rate scale it
# is a meaningful step between green and orange rather than a colour cast.
PRECIP_COLORS = [
    '#9fe8ff',  # 0.01 – 0.02  trace
    '#4fc0f0',  # 0.02 – 0.05  very light
    '#1878d0',  # 0.05 – 0.10  light
    '#17a92a',  # 0.10 – 0.20  light-moderate
    '#4fd12a',  # 0.20 – 0.40  moderate
    '#ffe000',  # 0.40 – 0.75  moderate-heavy
    '#ff9000',  # 0.75 – 1.25  heavy
    '#ff3000',  # 1.25 – 2.00  very heavy
    '#c00040',  # 2.00 – 3.00  extreme
    '#ff40ff',  # 3.00 – 4.00+ exceptional
]

# Opacity ramp — trace rain stays translucent so the basemap reads through,
# heavy cores are nearly opaque.
PRECIP_ALPHAS = np.linspace(0.45, 0.95, len(PRECIP_COLORS))


def _hex_to_rgb(h):
    """'#rrggbb' → (r, g, b) floats in [0, 1]."""
    h = h.lstrip('#')
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


# ---------------------------------------------------------------------------
# Experimental IR → precipitation rate estimator
# ---------------------------------------------------------------------------
# The core curve is the GOES Auto-Estimator power-law regression of Vicente,
# Scofield & Menzel (1998), which fitted 10.7 µm cloud-top brightness
# temperature against radar-derived rain rates:
#
#     R = 1.1183e11 · exp(-3.6382e-2 · T^1.2)      [mm/hr, T in K]
#
# AHI Band 13 (10.4 µm) is the Himawari equivalent of the GOES-8 10.7 µm
# channel the regression was built on.
AE_A = 1.1183e11
AE_B = 3.6382e-2
AE_P = 1.2


def _auto_estimator_rate(bt):
    """Bare Vicente et al. (1998) Auto-Estimator curve, mm/hr."""
    # Clamp the input so the exponential can't overflow on bad/fill pixels.
    bt = np.clip(bt, 180.0, 330.0)
    return AE_A * np.exp(-AE_B * np.power(bt, AE_P))


def _texture_factor(bt13, km_per_px):
    """Convective-core vs. cirrus-anvil weighting from cloud-top texture.

    The raw regression assigns rain to any cold pixel, so a wide cirrus anvil
    blooms into a huge false rain shield.  The Hydro-Estimator's fix — adopted
    here — compares each pixel to the mean temperature of the cloudy pixels
    around it.  Actively growing convective cores are locally *colder* than
    their surroundings; anvil that has spread downwind is locally *warmer*.

    The multiplier only ever suppresses (max 1.0): a colder-than-average pixel
    is already assigned more rain by the temperature curve itself, so boosting
    it again for being cold relative to its neighbours would double-count the
    same signal.  Returns ~0.1 for warm-relative anvil rising to 1.0 at and
    below the local cloud mean.
    """
    size = int(round(PRECIP_TEXTURE_KM / max(km_per_px, 0.5)))
    size = max(3, size | 1)  # odd window, at least 3 px

    cloud = (bt13 < PRECIP_CLOUD_MAX_K).astype(np.float32)
    # Mean over cloudy pixels only — clear-sky warmth must not dilute the mean.
    bt_sum = uniform_filter(np.where(cloud > 0, bt13, 0.0).astype(np.float32), size=size)
    bt_cnt = uniform_filter(cloud, size=size)
    local_mean = np.where(bt_cnt > 1e-6, bt_sum / np.maximum(bt_cnt, 1e-6), bt13)

    # Positive anomaly = colder than surroundings = likely active core
    anomaly = local_mean - bt13
    return np.interp(anomaly, [-6.0, -2.0, 0.0],
                              [0.10, 0.55, 1.00]).astype(np.float32)


def _wv_factor(bt13, bt_wv):
    """Deep-convection screen from the water-vapour minus IR difference.

    When a cloud top reaches the upper troposphere both Band 9 and Band 13 see
    the same surface, so the difference approaches zero (and goes slightly
    positive for overshooting tops that punch into the stratospheric
    inversion).  Either side of that narrow window means no deep convection:

      • Strongly negative — Band 9 still senses cold upper-tropospheric
        moisture well above a mid-level or thin cloud, so it reads far colder
        than the window channel.  Cold-but-dry cirrus lands here.
      • Strongly positive — the window channel is seeing something colder than
        the mid-troposphere itself, which no cloud top does.  That is a cold
        *surface*: sea ice or high terrain.  Without this branch the Antarctic
        margin inside this domain scores as continent-sized convection.
    """
    diff = bt_wv - bt13
    return np.interp(diff, [-35.0, -20.0, -8.0, 0.0, 5.0, 18.0],
                           [0.05, 0.25, 0.85, 1.00, 1.00, 0.00]).astype(np.float32)


def _split_window_factor(bt13, bt15):
    """Thin-cirrus screen from the split-window difference (Band 13 − Band 15).

    The 12.4 µm window channel is more strongly absorbing than the 10.4 µm
    one, so a semi-transparent cloud — which lets both channels see some of
    the warm surface below — produces a clearly positive difference.  An
    optically thick raining cloud is a near-blackbody in both channels, so the
    difference collapses towards zero.

    This is the classic discriminator for exactly the population the estimator
    is most prone to over-reading: cold, high, thin cirrus blown downwind of a
    storm that looks identical to the raining core in a single IR channel.
    """
    btd = bt13 - bt15
    return np.interp(btd, [1.5, 3.0, 6.5],
                          [1.00, 0.70, 0.10]).astype(np.float32)


def _view_angle_factor(x, y):
    """Taper the estimate to zero towards the limb of the disk.

    Near the edge of the full disk each pixel is smeared over a huge, highly
    oblique footprint and the line of sight passes through so much atmosphere
    that cloud-top temperatures are badly biased.  Beyond roughly 65° of
    satellite zenith angle the retrieval is not usable, which is also where
    the radial sampling artefacts appear.

    x, y are projection coordinates in metres (scan angle × satellite height).
    """
    # Angular distance from nadir along the scan, in radians.  Broadcast the
    # 1-D axes rather than meshgrid — on a 2750² grid that saves allocating
    # two full float64 copies.
    xs = np.asarray(x, dtype=np.float32)[None, :] / SAT_HEIGHT
    ys = np.asarray(y, dtype=np.float32)[:, None] / SAT_HEIGHT
    scan = np.sqrt(xs * xs + ys * ys)

    # Law of sines on the satellite–Earth-centre–pixel triangle:
    #   sin(scan) = (R_e / (R_e + h)) · sin(zenith)
    ratio = (EARTH_RADIUS + SAT_HEIGHT) / EARTH_RADIUS
    s = np.clip(np.sin(scan) * ratio, -1.0, 1.0)  # >1 would be off-earth
    zenith = np.degrees(np.arcsin(s))

    return np.interp(zenith, [65.0, 75.0], [1.0, 0.0]).astype(np.float32)


def _cooling_factor(bt13, bt13_prev, minutes):
    """Growth-rate weighting from the cloud-top temperature trend.

    A cooling top is a rising, still-building updraught and rains hardest; a
    warming top is a collapsing or spreading anvil where rainfall is already
    tapering off.  A single IR image cannot tell those two apart even though
    they look identical in temperature — comparing two scans can, and it is
    the largest single skill gain available from infrared alone.

    Weighted towards growth rather than symmetric about zero: sustained deep
    convection cools its top continuously while it builds, so a top that has
    stopped cooling is usually already past its peak.
    """
    if bt13_prev is None or minutes <= 0 or bt13_prev.shape != bt13.shape:
        return 1.0

    # Normalise the trend to K per 15 minutes so the thresholds below are
    # independent of the actual gap between the two scans.
    trend = (bt13 - bt13_prev) * (15.0 / minutes)
    return np.interp(trend, [-8.0, -2.0, 0.0, 2.0, 6.0],
                            [1.20, 1.05, 0.85, 0.50, 0.20]).astype(np.float32)


def estimate_precip_rate(bt13, bt_wv=None, bt13_prev=None,
                         prev_minutes=0, km_per_px=2.0, x=None, y=None,
                         bt_split=None):
    """Estimate surface rain rate (in/hr) from AHI infrared brightness temps.

    EXPERIMENTAL.  Infrared sees cloud *tops*, not raindrops, so this is an
    inference chain rather than a measurement.  Known limitations:

      • Warm rain from shallow maritime clouds is invisible — those tops are
        warmer than PRECIP_CLOUD_MAX_K and are scored as zero.  That matters
        more in this domain than over the continental US, because the warm
        pool west of the date line produces a great deal of it.
      • The result is therefore biased towards the convective part of the rain
        field.  Light stratiform rain falling from tops in the 240–260 K range
        is scored below the display threshold and simply disappears, so the
        rain area is too small and the surviving area skews heavy.  Read this
        as "where is the deep convection and roughly how hard is it", not as a
        complete rainfall map.
      • Cold cirrus that is not raining can still be scored as rain when the
        texture and water-vapour screens fail to catch it.
      • Himawari-9 carries no lightning mapper, so unlike the GOES version of
        this product there is no independent confirmation of a vigorous
        ice-phase updraught; the screens here are infrared-only.
      • No parallax correction, so cold tops are displaced from the rain
        beneath them by tens of km at high latitudes and near the limb.
      • Nothing is estimated beyond ~75° satellite zenith angle, so the outer
        ring of the disk is intentionally blank.
      • The Auto-Estimator regression was fitted over the continental US in
        summer; it over-predicts in dry environments and under-predicts in
        moist tropical ones.  The operational Hydro-Estimator corrects this
        with model precipitable water and RH, which are not used here.

    bt13        – Band 13 (10.4 µm) brightness temperature, K
    bt_wv       – optional Band 9 (6.9 µm) brightness temperature, K
    bt13_prev   – optional earlier Band 13 scan on the same grid, K
    prev_minutes– minutes between bt13_prev and bt13
    km_per_px   – ground sampling distance, used to size the texture window
    x, y        – optional projection coordinates (m) enabling the limb taper
    bt_split    – optional Band 15 (12.4 µm) brightness temperature, K
    """
    bt13 = np.asarray(bt13, dtype=np.float32)
    # Off-earth / fill pixels become "warm" so they score as no rain.
    valid = np.isfinite(bt13)
    bt13_f = np.where(valid, bt13, 330.0)

    def _match(arr, fill=330.0):
        """Coerce a companion band onto the Band 13 grid."""
        a = np.asarray(arr, dtype=np.float32)
        a = np.where(np.isfinite(a), a, fill)
        if a.shape != bt13_f.shape:
            a = zoom(a, (bt13_f.shape[0] / a.shape[0],
                         bt13_f.shape[1] / a.shape[1]), order=1)
        return a

    screen = _texture_factor(bt13_f, km_per_px)

    if bt_wv is not None:
        screen = screen * _wv_factor(bt13_f, _match(bt_wv))

    if bt_split is not None:
        screen = screen * _split_window_factor(bt13_f, _match(bt_split))

    if bt13_prev is not None:
        screen = screen * _cooling_factor(bt13_f, _match(bt13_prev), prev_minutes)

    rate = _auto_estimator_rate(bt13_f) * screen

    # Applied last: this is a data-quality mask rather than a convection
    # screen, so it is not something the other terms are allowed to override.
    if x is not None and y is not None:
        rate = rate * _view_angle_factor(x, y)

    # Nothing warmer than the cloud threshold precipitates in this scheme.
    rate = np.where(bt13_f > PRECIP_CLOUD_MAX_K, 0.0, rate)
    rate = np.where(valid, rate, 0.0)

    # The regression is published in mm/hr; the product reports inches.
    rate = rate / MM_PER_INCH

    rate = np.clip(np.nan_to_num(rate, nan=0.0, posinf=PRECIP_MAX_RATE),
                   0.0, PRECIP_MAX_RATE)
    rate = np.where(rate < PRECIP_MIN_RATE, 0.0, rate)
    return rate.astype(np.float32)


def _precip_rgba(rate):
    """Map a rain-rate field to an RGBA image using the discrete scale."""
    colors = np.array([_hex_to_rgb(c) for c in PRECIP_COLORS], dtype=np.float32)
    alphas = PRECIP_ALPHAS.astype(np.float32)

    # digitize returns 0 below the first edge, so subtract 1 to index bins.
    idx = np.digitize(rate, PRECIP_LEVELS) - 1
    has_rain = idx >= 0
    idx = np.clip(idx, 0, len(PRECIP_COLORS) - 1)

    rgb = colors[idx]
    a   = np.where(has_rain, alphas[idx], 0.0)
    return np.dstack([rgb, a[..., None]]).astype(np.float32)


def _precip_stats(rate):
    """One-line coverage summary, useful for spotting a broken run in the log."""
    raining = rate >= PRECIP_MIN_RATE
    n = int(raining.sum())
    if n == 0:
        return "no precipitating pixels"
    return (f"{100.0 * n / rate.size:.2f}% of pixels raining, "
            f"max {float(rate.max()):.2f} in/hr, "
            f"mean-where-raining {float(rate[raining].mean()):.3f} in/hr")


def _km_per_pixel(x):
    """Ground sampling distance in km from the projection x coordinates.

    x is scan angle × satellite height, so the spacing equals the nadir ground
    resolution of the (possibly downsampled) grid actually being rendered.
    Pixels grow towards the limb; this is the nadir value.
    """
    if x is None or len(x) < 2:
        return 2.0
    return abs(float(x[1] - x[0])) / 1000.0


# ---------------------------------------------------------------------------
# Frame management
# ---------------------------------------------------------------------------

def shift_frames(product_base):
    """Shift existing frames back one slot to make room for a new _00 frame.

    _00 is always the newest frame; _{MAX_FRAMES-1} is the oldest.
    When the buffer is full the oldest frame is deleted before shifting.
    A legacy single-file (product.png) is migrated to the oldest slot on
    the first call so no historical imagery is lost.
    """
    legacy   = os.path.join(OUTPUT_DIR, f'{product_base}.png')
    frame_00 = os.path.join(OUTPUT_DIR, f'{product_base}_00.png')

    # One-time migration: seed the oldest slot with the pre-frame-buffer image
    if os.path.exists(legacy) and not os.path.exists(frame_00):
        seed = os.path.join(OUTPUT_DIR, f'{product_base}_{MAX_FRAMES - 1:02d}.png')
        os.rename(legacy, seed)
        print(f"  Migrated legacy {product_base}.png → {os.path.basename(seed)}")

    # Count how many frame files currently exist
    n_existing = sum(
        1 for i in range(MAX_FRAMES)
        if os.path.exists(os.path.join(OUTPUT_DIR, f'{product_base}_{i:02d}.png'))
    )

    # Drop the oldest frame only when the buffer is already at capacity
    if n_existing >= MAX_FRAMES:
        oldest = os.path.join(OUTPUT_DIR, f'{product_base}_{MAX_FRAMES - 1:02d}.png')
        if os.path.exists(oldest):
            os.remove(oldest)

    # Shift _08→_09, _07→_08, …, _00→_01
    for i in range(MAX_FRAMES - 2, -1, -1):
        src = os.path.join(OUTPUT_DIR, f'{product_base}_{i:02d}.png')
        dst = os.path.join(OUTPUT_DIR, f'{product_base}_{i + 1:02d}.png')
        if os.path.exists(src):
            os.rename(src, dst)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_png(output_path, dpi=150):
    """Save the current figure as a size-optimised transparent PNG, then close it.

    Every frame is committed straight into the repository on a 10-minute
    cycle, so bytes written here become repository growth.  Writing an 8-bit
    palette PNG instead of full RGBA cuts each frame to roughly a quarter of
    its size.

    The colour-mapped products (IR, Water Vapor, Visible, Precip) draw from a
    256-entry colormap, so they contain few enough distinct colours to be
    re-indexed *exactly* — those frames are bit-for-bit identical to the RGBA
    render, just stored more compactly.  GeoColor is a true-colour composite
    with tens of thousands of colours and falls back to a 256-colour adaptive
    palette, which is visually indistinguishable at this resolution.
    """
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=dpi, transparent=True)
    plt.close()
    buf.seek(0)

    img = Image.open(buf).convert('RGBA')
    arr = np.ascontiguousarray(np.array(img, dtype=np.uint8))
    h, w = arr.shape[:2]

    # Pack RGBA into one uint32 per pixel so the unique/index pass is a cheap
    # 1-D sort rather than a lexsort over millions of 4-element rows.
    packed = arr.reshape(-1, 4).view(np.uint32).reshape(-1)
    uniq, idx = np.unique(packed, return_inverse=True)

    if len(uniq) <= 256:
        rgba = uniq.view(np.uint8).reshape(-1, 4)
        out = Image.fromarray(idx.astype(np.uint8).reshape(h, w), mode='P')
        palette = rgba[:, :3].reshape(-1).tolist()
        out.putpalette(palette + [0] * (768 - len(palette)))
        out.save(output_path, format='PNG', optimize=True,
                 transparency=bytes(rgba[:, 3].tolist()))
    else:
        img.quantize(colors=256, method=Image.FASTOCTREE).save(
            output_path, format='PNG', optimize=True)


def get_latest_scan_time(s3_client):
    """Find the most recent available Himawari-9 scan (every 10 minutes).

    Checks for the presence of Band 3 (visible) files specifically, not just
    any file in the prefix.  This avoids a race condition where Band 01 has
    been uploaded to S3 but Band 03 (and others) have not yet arrived — the
    uploader streams bands in numerical order, so a partial scan prefix is
    common for the most-recent time slot.

    Returns (year, month, day, hhmm) as strings, or None if no data found
    within the past 6 hours.
    """
    now = datetime.now(timezone.utc)
    for offset in range(0, 6 * 60, 10):  # search back up to 6 hours in 10-min steps
        t = now - timedelta(minutes=offset)
        # Round down to the nearest 10-minute boundary
        t = t.replace(minute=(t.minute // 10) * 10, second=0, microsecond=0)
        year  = t.strftime('%Y')
        month = t.strftime('%m')
        day   = t.strftime('%d')
        hhmm  = t.strftime('%H%M')
        # Use a Band-03-specific prefix so we only accept scans where the
        # visible band is fully available (a reliable proxy for a complete scan).
        b03_prefix = (f'AHI-L1b-FLDK/{year}/{month}/{day}/{hhmm}/'
                      f'HS_H09_{year}{month}{day}_{hhmm}_B03_')
        try:
            resp = s3_client.list_objects_v2(Bucket=BUCKET, Prefix=b03_prefix, MaxKeys=1)
            if resp.get('Contents'):
                print(f'  Latest scan: {year}/{month}/{day} {hhmm} UTC')
                return year, month, day, hhmm
        except Exception as e:
            print(f'  Warning scanning {b03_prefix}: {e}')
    return None


def get_previous_scan_time(s3_client, scan_time, minutes=PRECIP_COOLING_MINUTES):
    """Find a Band 13 scan roughly *minutes* before *scan_time*.

    Searches backwards from the target time in 10-minute steps (the AHI
    full-disk cadence) for up to an hour beyond it, so a missing scan degrades
    into a slightly longer baseline rather than losing the term entirely.

    Returns ((year, month, day, hhmm), gap_minutes) or (None, 0).
    """
    if minutes <= 0:
        return None, 0

    year, month, day, hhmm = scan_time
    current = datetime(int(year), int(month), int(day),
                       int(hhmm[:2]), int(hhmm[2:]), tzinfo=timezone.utc)

    for extra in range(0, 61, 10):
        t = current - timedelta(minutes=minutes + extra)
        y, mo, d, hm = (t.strftime('%Y'), t.strftime('%m'),
                        t.strftime('%d'), t.strftime('%H%M'))
        prefix = (f'AHI-L1b-FLDK/{y}/{mo}/{d}/{hm}/'
                  f'HS_H09_{y}{mo}{d}_{hm}_B13_')
        try:
            resp = s3_client.list_objects_v2(Bucket=BUCKET, Prefix=prefix, MaxKeys=1)
        except Exception as e:
            print(f'  Warning scanning {prefix}: {e}')
            continue
        if resp.get('Contents'):
            gap = (current - t).total_seconds() / 60.0
            print(f'  Lookback scan: {y}/{mo}/{d} {hm} UTC ({gap:.0f} min before current)')
            return (y, mo, d, hm), gap

    print('  Note: no earlier Band 13 scan found; skipping cooling-rate term.')
    return None, 0


def _make_figure():
    """Create a matplotlib figure in Web Mercator projection.

    Rendering in Mercator (rather than PlateCarree) ensures the PNG pixels
    map linearly onto Leaflet's tile coordinate system, so the L.imageOverlay
    is not squished and geographic coordinates align correctly.
    """
    # Compute Mercator Y extent to derive the correct figure aspect ratio.
    # Formula: y = R * ln(tan(π/4 + φ/2))  (standard Web Mercator / EPSG:3857)
    R = 6_378_137.0  # WGS-84 semi-major axis (metres)
    lat_min_rad = np.radians(EXTENT[2])
    lat_max_rad = np.radians(EXTENT[3])
    merc_y_max = R * np.log(np.tan(np.pi / 4.0 + lat_max_rad / 2.0))
    merc_y_min = R * np.log(np.tan(np.pi / 4.0 + lat_min_rad / 2.0))
    merc_y_range = merc_y_max - merc_y_min          # ~16.8 M m for 60 °S–60 °N
    merc_x_range = np.radians(EXTENT[1] - EXTENT[0]) * R  # ~13.4 M m for 120 ° lon

    fig_width  = 12.0
    fig_height = fig_width * merc_y_range / merc_x_range  # ≈ 15.1 in for this domain

    fig = plt.figure(figsize=(fig_width, fig_height))
    # Mercator centred on SAT_LON keeps the antimeridian crossing away from the
    # map edge and matches the Web Mercator tiles used by Leaflet.
    crs = ccrs.Mercator(central_longitude=SAT_LON)
    ax  = fig.add_axes([0, 0, 1, 1], projection=crs)
    # Use a PlateCarree shifted to SAT_LON so the 200 °E coordinate stays
    # within [−180, 180] and Cartopy handles it unambiguously.
    geog_crs = ccrs.PlateCarree(central_longitude=SAT_LON)
    ax.set_extent(
        [EXTENT[0] - SAT_LON, EXTENT[1] - SAT_LON, EXTENT[2], EXTENT[3]],
        crs=geog_crs,
    )
    ax.set_axis_off()
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    return fig, ax


def _make_figure_sector(extent):
    """Create a matplotlib figure in Web Mercator for a cyclone sector.

    Same approach as _make_figure() but uses the supplied extent and a
    higher DPI / different figure size tuned for sector crops.
    """
    if extent[1] <= extent[0] or extent[3] <= extent[2]:
        raise ValueError(f'degenerate sector extent {extent}')

    R = 6_378_137.0
    lat_min_rad = np.radians(extent[2])
    lat_max_rad = np.radians(extent[3])
    merc_y_max = R * np.log(np.tan(np.pi / 4.0 + lat_max_rad / 2.0))
    merc_y_min = R * np.log(np.tan(np.pi / 4.0 + lat_min_rad / 2.0))
    merc_y_range = merc_y_max - merc_y_min
    merc_x_range = np.radians(extent[1] - extent[0]) * R

    fig_height = SECTOR_FIG_WIDTH * merc_y_range / merc_x_range

    fig = plt.figure(figsize=(SECTOR_FIG_WIDTH, fig_height))
    crs = ccrs.Mercator(central_longitude=SAT_LON)
    ax  = fig.add_axes([0, 0, 1, 1], projection=crs)
    geog_crs = ccrs.PlateCarree(central_longitude=SAT_LON)
    ax.set_extent(
        [extent[0] - SAT_LON, extent[1] - SAT_LON, extent[2], extent[3]],
        crs=geog_crs,
    )
    ax.set_axis_off()
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    return fig, ax


def fetch_active_storms():
    """Fetch active storm list from JTWC data URL.

    Returns a list of storm dicts (may be empty on any error).
    Each dict has at minimum: id, name, basin, current.lat/lon/wind_kt/
    classification/classification_label.
    """
    try:
        req = urllib.request.Request(STORMS_JSON_URL,
                                     headers={'User-Agent': 'HimawariSatViewer/1.0'})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
        storms = data.get('storms', [])
        print(f"  Fetched {len(storms)} active storm(s) from JTWC data.")
        return storms
    except Exception as e:
        print(f"  WARNING: Could not fetch storm data: {e}")
        return []


def _normalise_lon(lon):
    """Map a signed longitude onto the 0–360 convention EXTENT is written in.

    The JTWC feed reports longitude in [-180, 180], but the Himawari domain
    spans 80 °E to 200 °E (= 160 °W), so a storm east of the antimeridian
    arrives as a negative number that sorts *below* the domain's west edge.
    Adding 360 puts it back on the correct side of the disk.
    """
    if lon is None:
        return None
    lon = float(lon)
    if not np.isfinite(lon):
        return None
    while lon < EXTENT[0]:
        lon += 360.0
    while lon > EXTENT[0] + 360.0:
        lon -= 360.0
    return lon


def _sector_bounds(lat, lon):
    """Return [west, east, south, north] for a cyclone sector centred on lat/lon.

    Returns None when the storm does not fall inside the Himawari domain, or
    when clipping to EXTENT leaves a degenerate (zero- or negative-width) box.
    An East-Pacific storm at, say, 213 °E is outside this satellite's view
    entirely; previously it produced east < west, which propagated into a
    negative figure height and aborted the whole run.
    """
    if not (np.isfinite(lat) and np.isfinite(lon)):
        return None

    west  = max(lon - SECTOR_PAD_DEG, EXTENT[0])
    east  = min(lon + SECTOR_PAD_DEG, EXTENT[1])
    south = max(lat - SECTOR_PAD_DEG, EXTENT[2])
    north = min(lat + SECTOR_PAD_DEG, EXTENT[3])

    # Require a usable sliver of sector, not just a non-empty intersection:
    # a storm one degree inside the limb yields a box too thin to render.
    if east - west < MIN_SECTOR_DEG or north - south < MIN_SECTOR_DEG:
        return None
    return [west, east, south, north]


def _render_sector_band(arr, x, y, sat_proj, extent, product_base, colormap,
                        vmin, vmax, gamma=1.0):
    """Render a band array cropped to *extent* and save as a high-res sector PNG."""
    # Apply gamma before rendering
    if gamma != 1.0:
        normed = np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)
        arr    = np.power(normed, gamma) * (vmax - vmin) + vmin

    fig, ax = _make_figure_sector(extent)
    img_extent = (x[0], x[-1], y[-1], y[0])
    ax.imshow(arr, origin='upper', extent=img_extent,
              transform=sat_proj, cmap=colormap,
              vmin=vmin, vmax=vmax, interpolation='none')
    shift_frames(product_base)
    out = os.path.join(OUTPUT_DIR, f'{product_base}_00.png')
    _save_png(out, dpi=SECTOR_DPI)
    print(f"    Saved sector: {out}")


def _crop_band_to_sector(arr, x, y, sat_proj, bounds):
    """Crop a full-disk band array to just the pixels covering the sector.

    Converts the sector's lat/lon corners to geostationary x/y metres,
    then slices arr/x/y to that rectangle with a small pixel buffer.

    A 16° sector is ~13 % of the full disk in each axis, so the result
    is roughly 365×365 instead of 2 750×2 750 — about 56× fewer elements.
    This keeps Cartopy's reprojection buffers small and prevents OOM when
    rendering many sector products in one run.

    Returns (arr_crop, x_crop, y_crop); falls back to originals on error.
    """
    try:
        pc   = ccrs.PlateCarree()
        lons = np.array([bounds[0], bounds[0], bounds[1], bounds[1]])
        lats = np.array([bounds[2], bounds[3], bounds[2], bounds[3]])
        pts  = sat_proj.transform_points(pc, lons, lats)
        gx_min = np.nanmin(pts[:, 0])
        gx_max = np.nanmax(pts[:, 0])
        gy_min = np.nanmin(pts[:, 1])
        gy_max = np.nanmax(pts[:, 1])

        # x is ascending, y is descending
        ix = np.where((x >= gx_min) & (x <= gx_max))[0]
        iy = np.where((y >= gy_min) & (y <= gy_max))[0]
        if len(ix) == 0 or len(iy) == 0:
            return arr, x, y

        # Add a small pixel border so the sector edge isn't clipped
        PAD = 10
        ix_lo = max(0,       ix[0]  - PAD)
        ix_hi = min(len(x),  ix[-1] + PAD + 1)
        iy_lo = max(0,       iy[0]  - PAD)
        iy_hi = min(len(y),  iy[-1] + PAD + 1)

        return (arr[iy_lo:iy_hi, ix_lo:ix_hi],
                x[ix_lo:ix_hi],
                y[iy_lo:iy_hi])
    except Exception as e:
        print(f"    Warning: sector crop failed ({e}), using full array")
        return arr, x, y


def process_cyclone_sectors(s3_client, scan_time, storms):
    """Generate high-resolution sector images for each active cyclone.

    For every storm, five sector PNGs are produced (geocolor, visible,
    infrared, water_vapor, precip) using the full-native-resolution band data
    cropped to a ~16°×16° box centred on the storm.  A sectors_meta.json
    file is written to OUTPUT_DIR listing each sector's metadata and
    geographic bounds so the frontend can display the correct overlays.
    """
    if not storms:
        # Write an empty metadata file so the frontend can detect no sectors
        _write_sectors_meta([])
        return

    # Reuse the already-cached full-disk bands — no extra downloads needed.
    # All four bands were fetched during full-disk processing and are in
    # _band_cache at MAX_BAND_PX resolution.  Sectors render a small
    # geographic slice of this data, so matplotlib naturally shows more
    # detail per screen pixel than the full-disk view does.
    b1,  x1,  y1,  goes_proj = _download_band(s3_client, 1,  scan_time)
    b3,  x3,  y3,  _         = _download_band(s3_client, 3,  scan_time)
    b9,  x9,  y9,  _         = _download_band(s3_client, 9,  scan_time)
    b13, x13, y13, _         = _download_band(s3_client, 13, scan_time)

    # Precipitation inputs.  Every band involved is already in _band_cache from
    # the full-disk pass, so this resolves without new downloads; only the
    # lookback-scan lookup costs an S3 listing.
    precip_inputs = _precip_inputs(s3_client, scan_time)

    sectors_meta = []

    for storm in storms:
        sid   = storm.get('id', 'UNKNOWN')
        name  = storm.get('name', 'Unknown')
        cur   = storm.get('current', {})
        clat  = cur.get('lat')
        clon  = _normalise_lon(cur.get('lon'))

        if clat is None or clon is None:
            print(f"  Skipping sector for {sid}: no position data.")
            continue

        bounds = _sector_bounds(clat, clon)
        if bounds is None:
            # Typically an Atlantic or East-Pacific storm carried by the same
            # feed — outside Himawari's disk, so there is nothing to render.
            print(f"  Skipping sector for {name} ({sid}): "
                  f"{clat}°N, {clon}°E is outside the Himawari domain "
                  f"{EXTENT[0]}–{EXTENT[1]}°E, {EXTENT[2]}–{EXTENT[3]}°N.")
            continue

        try:
            print(f"\n--- Cyclone Sector: {name} ({sid})  [{clat}°N, {clon}°E] ---")
            print(f"    Sector bounds: W{bounds[0]} E{bounds[1]} S{bounds[2]} N{bounds[3]}")

            safe_id = sid.replace('/', '_')

            # Crop every band to the sector extent before rendering.
            # A 16° sector is ~13 % of the full disk per axis, so the cropped
            # arrays are ~365×365 instead of 2 750×2 750 — ~56× fewer elements.
            # This keeps Cartopy's reprojection buffers small and prevents OOM.
            b1s,  x1s,  y1s  = _crop_band_to_sector(b1,  x1,  y1,  goes_proj, bounds) if b1  is not None else (None, None, None)
            b3s,  x3s,  y3s  = _crop_band_to_sector(b3,  x3,  y3,  goes_proj, bounds) if b3  is not None else (None, None, None)
            b9s,  x9s,  y9s  = _crop_band_to_sector(b9,  x9,  y9,  goes_proj, bounds) if b9  is not None else (None, None, None)
            b13s, x13s, y13s = _crop_band_to_sector(b13, x13, y13, goes_proj, bounds) if b13 is not None else (None, None, None)

            if b1s is not None:
                print(f"    Cropped Band 1:  {b1.shape} → {b1s.shape}")
            if b3s is not None:
                print(f"    Cropped Band 3:  {b3.shape} → {b3s.shape}")
            if b13s is not None:
                print(f"    Cropped Band 13: {b13.shape} → {b13s.shape}")

            # ---- GeoColor sector ----
            if b1s is not None and b3s is not None:
                b1_f = np.nan_to_num(b1s, nan=0.0)
                b3_f = np.nan_to_num(b3s, nan=0.0)
                # If the two cropped bands have different shapes, resize to match
                if b3_f.shape != b1_f.shape:
                    b3_f = zoom(b3_f, (b1_f.shape[0]/b3_f.shape[0],
                                       b1_f.shape[1]/b3_f.shape[1]), order=1)
                mean_ref = float(np.nanmean(b3_f))
                if mean_ref > 0.05:  # daytime
                    R = np.power(np.clip(b3_f, 0, 1), 0.6)
                    G = np.power(np.clip(0.5*b3_f + 0.5*b1_f, 0, 1), 0.6)
                    B = np.power(np.clip(b1_f, 0, 1), 0.6)
                    if b13s is not None:
                        bt13_f = np.where(np.isnan(b13s), 320.0, b13s)
                        if bt13_f.shape != R.shape:
                            bt13_f = zoom(bt13_f, (R.shape[0]/bt13_f.shape[0],
                                                   R.shape[1]/bt13_f.shape[1]), order=1)
                        ce = np.clip((235.0 - bt13_f) / 45.0, 0.0, 1.0) * 0.5
                        R = np.clip(R + ce*(1-R), 0, 1)
                        G = np.clip(G + ce*(1-G), 0, 1)
                        B = np.clip(B + ce*(1-B), 0, 1)
                    rgb = np.dstack([R, G, B])
                    fig, ax = _make_figure_sector(bounds)
                    img_ext = (x1s[0], x1s[-1], y1s[-1], y1s[0])
                    ax.imshow(rgb, origin='upper', extent=img_ext,
                              transform=goes_proj, interpolation='none')
                else:  # nighttime — IR cloud composite
                    if b13s is not None:
                        bt13_n = np.where(np.isnan(b13s), 320.0, b13s)
                        co = np.clip((275.0 - bt13_n) / 55.0, 0.0, 1.0)
                        R2 = co * 0.80; G2 = co * 0.88; B2 = co * 1.00
                        A2 = np.clip(co * 1.5, 0.0, 1.0)
                        rgba = np.dstack([R2, G2, B2, A2]).astype(np.float32)
                        fig, ax = _make_figure_sector(bounds)
                        img_ext = (x13s[0], x13s[-1], y13s[-1], y13s[0])
                        ax.imshow(rgba, origin='upper', extent=img_ext,
                                  transform=goes_proj, interpolation='none')
                    else:
                        fig, ax = _make_figure_sector(bounds)

                pb = f'geocolor_sector_{safe_id}'
                shift_frames(pb)
                _save_png(os.path.join(OUTPUT_DIR, f'{pb}_00.png'), dpi=SECTOR_DPI)
                print(f"    Saved sector GeoColor.")

            # ---- Visible sector ----
            if b3s is not None:
                b3_vis = np.nan_to_num(b3s, nan=0.0)
                _render_sector_band(b3_vis, x3s, y3s, goes_proj, bounds,
                                    f'visible_sector_{safe_id}', 'gray', 0.0, 1.0, gamma=0.6)

            # ---- Infrared sector ----
            if b13s is not None:
                _render_sector_band(b13s, x13s, y13s, goes_proj, bounds,
                                    f'infrared_sector_{safe_id}', _ir_colormap(), 190, 310)

            # ---- Water vapor sector ----
            if b9s is not None:
                _render_sector_band(b9s, x9s, y9s, goes_proj, bounds,
                                    f'water_vapor_sector_{safe_id}', _wv_colormap(), 195, 280)

            # ---- Precipitation rate sector ----
            _render_sector_precip(precip_inputs, bounds, f'precip_sector_{safe_id}')

            # Free cropped arrays and force GC before next storm
            del b1s, b3s, b9s, b13s
            gc.collect()
        except Exception as e:
            # One unrenderable storm must not cost the other sectors, nor
            # the full-disk products that have already been written.
            print(f"  WARNING: sector rendering failed for {name} ({sid}): {e}")
            import traceback
            traceback.print_exc()
            plt.close('all')  # drop any half-built figure before the next storm
            continue

        sectors_meta.append({
            'id':                   sid,
            'safe_id':              safe_id,
            'name':                 name,
            'basin':                storm.get('basin', ''),
            'basin_name':           storm.get('basin_name', ''),
            'classification':       cur.get('classification', ''),
            'classification_label': cur.get('classification_label', ''),
            'wind_kt':              cur.get('wind_kt', 0),
            'wind_mph':             cur.get('wind_mph', 0),
            'pressure_mb':          cur.get('pressure_mb', 0),
            'lat':                  clat,
            'lon':                  clon,
            'bounds':               bounds,   # [west, east, south, north]
        })

    _write_sectors_meta(sectors_meta)


def _write_sectors_meta(sectors_meta):
    """Write sectors_meta.json so the frontend knows which sectors exist."""
    meta = {
        'generated': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'sectors':   sectors_meta,
    }
    path = os.path.join(OUTPUT_DIR, 'sectors_meta.json')
    with open(path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"  Sectors metadata written: {path}  ({len(sectors_meta)} sector(s))")


def _download_band(s3_client, band_num, scan_time, max_px=MAX_BAND_PX):
    """Download all HSD segments for a Himawari-9 AHI band and load with satpy.

    max_px controls the maximum pixels-per-dimension after loading.  Pass
    None to skip downsampling entirely (native resolution).  Calls with
    different max_px values are cached separately so high-res sector
    downloads don't evict the lower-res full-disk versions.

    Returns (data_array, x_metres, y_metres, sat_proj).
    data_array contains calibrated float32 values (reflectance or BT) with NaN
    for off-Earth pixels.  Returns (None, None, None, None) on any failure.
    """
    cache_key = (band_num, scan_time, max_px)
    if cache_key in _band_cache:
        print(f'  Band {band_num}: using cached data (max_px={max_px})')
        return _band_cache[cache_key]

    year, month, day, hhmm = scan_time
    prefix   = f'AHI-L1b-FLDK/{year}/{month}/{day}/{hhmm}/'
    band_tag = f'_B{band_num:02d}_'

    print(f'  Downloading Band {band_num}...')
    try:
        resp = s3_client.list_objects_v2(Bucket=BUCKET, Prefix=prefix)
        keys = sorted([
            obj['Key'] for obj in resp.get('Contents', [])
            if band_tag in obj['Key']
        ])
    except Exception as e:
        print(f'  ERROR listing {prefix}: {e}')
        return None, None, None, None

    if not keys:
        print(f'  ERROR: No segments found for B{band_num:02d} in {prefix}')
        return None, None, None, None

    tmpdir = f'/tmp/ahi_B{band_num:02d}'
    os.makedirs(tmpdir, exist_ok=True)
    local_files = []
    try:
        for key in keys:
            fname = key.split('/')[-1]
            local = os.path.join(tmpdir, fname)
            s3_client.download_file(BUCKET, key, local)
            local_files.append(local)
        print(f'    {len(local_files)} segments downloaded')

        band_name = f'B{band_num:02d}'
        scn = Scene(filenames=local_files, reader='ahi_hsd')
        scn.load([band_name])
        ds      = scn[band_name]
        arr     = ds.values.astype(np.float32)

        # Satpy calibrates AHI visible/near-IR channels (B01–B06) to
        # reflectance, but may return percent (0–100) rather than fraction
        # (0–1) depending on the satpy version.  All downstream code
        # (vmin/vmax, SZA correction, gamma) assumes the fraction form.
        # Detect via the dataset's units attribute and normalise if needed.
        if band_num <= 6:
            units = ds.attrs.get('units', '')
            if units == '%':
                arr /= 100.0
                print(f'  Band {band_num}: converted from percent reflectance to fraction [0, 1]')
            elif np.nanmax(arr) > 2.0:
                # Fallback: units tag missing but values clearly in percent range
                arr /= 100.0
                print(f'  Band {band_num}: normalised to fraction [0, 1] (units tag was "{units}")')

        area    = ds.attrs['area']
        left, bottom, right, top = area.area_extent
        n_rows, n_cols = arr.shape

        # Downsample high-resolution bands to stay within max_px.
        # max_px=None means no downsampling (native resolution for sectors).
        if max_px is not None and max(n_rows, n_cols) > max_px:
            scale = max_px / max(n_rows, n_cols)
            arr   = zoom(arr, scale, order=1)
            n_rows, n_cols = arr.shape

        x = np.linspace(left, right, n_cols)
        y = np.linspace(top, bottom, n_rows)  # decreasing: top → bottom

        sat_h    = float(area.proj_dict.get('h', 35785863))
        sat_proj = ccrs.Geostationary(central_longitude=SAT_LON, satellite_height=sat_h)

        v_min, v_max = float(np.nanmin(arr)), float(np.nanmax(arr))
        print(f'  Band {band_num}: shape={arr.shape}, range=[{v_min:.2f}, {v_max:.2f}]')

        result = (arr, x, y, sat_proj)
        _band_cache[cache_key] = result
        return result

    except Exception as e:
        print(f'  ERROR loading Band {band_num}: {e}')
        import traceback
        traceback.print_exc()
        return None, None, None, None

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Solar zenith angle correction
# ---------------------------------------------------------------------------

def _compute_cos_sza(x, y, sat_proj, scan_time):
    """Return cos(solar zenith angle) for each pixel in the satellite data grid.

    Computation is performed on a coarse sub-sampled grid (≈ 350×350 points)
    using Cartopy's geostationary→PlateCarree transform, then bilinearly
    upsampled to the full data resolution.  This avoids converting millions
    of points through Cartopy while keeping the smooth SZA field accurate.

    Returns a float32 array of shape (len(y), len(x)).
    Night-side and off-disk pixels are 0.0; zenith (sun directly overhead) is 1.0.
    """
    year, month, day, hhmm = scan_time
    dt  = datetime(int(year), int(month), int(day),
                   int(hhmm[:2]), int(hhmm[2:]), tzinfo=timezone.utc)
    utc_hours = dt.hour + dt.minute / 60.0

    # Solar declination — Spencer (1971) approximation (±0.3° accuracy)
    doy = dt.timetuple().tm_yday
    B   = 2.0 * np.pi * (doy - 1) / 365.0
    dec = (0.006918
           - 0.399912 * np.cos(B)    + 0.070257 * np.sin(B)
           - 0.006758 * np.cos(2*B)  + 0.000907 * np.sin(2*B))

    # Equation of time in hours (same source)
    eot = ((0.000075
            + 0.001868 * np.cos(B)   - 0.032077 * np.sin(B)
            - 0.014615 * np.cos(2*B) - 0.04089  * np.sin(2*B))
           * (12.0 / np.pi))

    # Work on a coarse grid for speed
    stride  = max(1, len(x) // 350)
    x_ds    = x[::stride]
    y_ds    = y[::stride]
    xx, yy  = np.meshgrid(x_ds, y_ds)

    # Geostationary → geographic (PlateCarree)
    pc   = ccrs.PlateCarree()
    pts  = pc.transform_points(sat_proj, xx.ravel(), yy.ravel())
    lons = pts[:, 0].reshape(yy.shape)   # degrees east
    lats = pts[:, 1].reshape(yy.shape)   # degrees north

    # Hour angle at each pixel
    tst = utc_hours + lons / 15.0 + eot          # true solar time (hours)
    ha  = np.radians((tst - 12.0) * 15.0)        # hour angle (radians)

    lat_r   = np.radians(lats)
    cos_sza = (np.sin(lat_r) * np.sin(dec)
               + np.cos(lat_r) * np.cos(dec) * np.cos(ha))

    # Night-side and off-disk (NaN lon/lat) → 0
    cos_sza = np.where(np.isnan(cos_sza), 0.0, cos_sza)
    cos_sza = np.maximum(cos_sza, 0.0).astype(np.float32)

    # Upsample back to full data resolution with bilinear interpolation
    if cos_sza.shape != (len(y), len(x)):
        scale_y = len(y) / cos_sza.shape[0]
        scale_x = len(x) / cos_sza.shape[1]
        cos_sza = zoom(cos_sza, (scale_y, scale_x), order=1)

    return np.clip(cos_sza, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Single-band renderer
# ---------------------------------------------------------------------------

def process_goes_band(s3_client, band, scan_time, output_filename, colormap,
                      vmin, vmax, gamma=1.0, sun_correct=False):
    """Download and render a single Himawari-9 AHI band as a transparent PNG.

    gamma       – power-law correction applied after normalising to [0, 1].
                  gamma < 1 brightens the image (e.g. 0.5 = square-root).
    sun_correct – if True, multiply the data by cos(solar zenith angle) before
                  gamma.  This converts satellite-measured apparent reflectance
                  (which blows up near the terminator because it is divided by
                  cos SZA) back to approximate physical albedo, eliminating the
                  blinding-white terminator wave.
    """
    print(f"\n--- Band {band}: {output_filename} ---")

    cmi_data, x, y, sat_proj = _download_band(s3_client, band, scan_time)
    if cmi_data is None:
        print(f"  Skipping {output_filename}.")
        return

    # Solar zenith correction: restore physical albedo from apparent reflectance
    if sun_correct:
        cos_sza  = _compute_cos_sza(x, y, sat_proj, scan_time)
        cmi_data = cmi_data * cos_sza
        print(f"  Solar zenith correction applied; mean cos(SZA) = {float(cos_sza.mean()):.3f}")

    # Apply gamma correction if requested (normalise → gamma → restore range)
    if gamma != 1.0:
        normed   = np.clip((cmi_data - vmin) / (vmax - vmin), 0.0, 1.0)
        cmi_data = np.power(normed, gamma) * (vmax - vmin) + vmin

    fig, ax = _make_figure()

    # imshow is much faster than pcolormesh for regular-grid geostationary data
    img_extent = (x[0], x[-1], y[-1], y[0])  # (left, right, bottom, top)
    ax.imshow(
        cmi_data, origin='upper', extent=img_extent,
        transform=sat_proj,
        cmap=colormap, vmin=vmin, vmax=vmax, interpolation='none',
    )

    product_base = output_filename.replace('.png', '')
    shift_frames(product_base)
    output_path = os.path.join(OUTPUT_DIR, f'{product_base}_00.png')
    _save_png(output_path)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# GeoColor composite (day / night)
# ---------------------------------------------------------------------------

def process_geocolor(s3_client, scan_time):
    """GeoColor RGB composite.

    Daytime  – pseudo-natural colour from Bands 1 and 3 with gamma correction.
    Nighttime – IR cloud layer (Band 13) blended with city-lights proxy
                (Band 7 minus thermal background) on a transparent background.
    """
    print(f"\n--- GeoColor RGB Composite ---")

    # Fetch the two visible bands needed for the RGB composite
    b1, x1, y1, goes_proj = _download_band(s3_client, 1, scan_time)
    if b1 is None:
        print("  ERROR: Missing Band 1. Skipping GeoColor.")
        return

    b2, x2, y2, _ = _download_band(s3_client, 3, scan_time)
    if b2 is None:
        print("  ERROR: Missing Band 3. Skipping GeoColor.")
        return

    # Band 3 is 0.5 km (2× resolution) – downsample to match Band 1 (1 km)
    b2 = np.nan_to_num(b2, nan=0.0)
    if b2.shape != b1.shape:
        zy = b1.shape[0] / b2.shape[0]
        zx = b1.shape[1] / b2.shape[1]
        b2 = zoom(b2, (zy, zx), order=1)

    b1 = np.nan_to_num(b1, nan=0.0)

    # Determine day vs night: at night Band 3 visible reflectance ≈ 0
    mean_ref = float(np.nanmean(b2))
    is_daytime = mean_ref > 0.05
    print(f"  Band 3 mean reflectance: {mean_ref:.4f}  →  {'DAYTIME' if is_daytime else 'NIGHTTIME'}")

    if is_daytime:
        # Fetch Band 13 for cloud-top enhancement (failure is non-fatal)
        bt13, *_ = _download_band(s3_client, 13, scan_time)
        _render_geocolor_day(b1, b2, x1, y1, goes_proj, bt13, scan_time=scan_time)
    else:
        _render_geocolor_night(s3_client, scan_time)


def _render_geocolor_day(b1, b2, x, y, goes_proj, bt13=None, scan_time=None):
    """Pseudo-natural colour composite for daytime.

    b1 = Band 1 (0.47 µm blue), b2 = Band 3 (0.64 µm red).
    bt13 is an optional Band 13 brightness-temperature array (same spatial
    footprint after resampling).  When provided, very cold cloud tops are
    blended towards bright white so deep convective anvils are clearly
    visible against land/ocean backgrounds.
    scan_time is required for the solar zenith angle correction.
    """
    # Solar zenith angle correction: convert apparent reflectance → physical
    # albedo so that the near-terminator region fades naturally rather than
    # blowing out to white.
    if scan_time is not None:
        cos_sza = _compute_cos_sza(x, y, goes_proj, scan_time)
        b1 = b1 * cos_sza
        b2 = b2 * cos_sza

    R = np.clip(b2, 0, 1)
    # Synthetic green: average of red and blue channels only.
    # Omitting NIR (Band 3 / 0.86 µm) prevents the yellow cast that NIR
    # introduces over vegetated and arid land surfaces.
    G = np.clip(0.5 * b2 + 0.5 * b1, 0, 1)
    B = np.clip(b1, 0, 1)

    # Gamma correction — with SZA-corrected physical albedos the data is in
    # the natural [0, 1] range so gamma=0.6 gives a pleasant stretch without
    # blowing out clouds or ocean sun glint.
    gamma = 0.6
    R = np.power(R, gamma)
    G = np.power(G, gamma)
    B = np.power(B, gamma)

    # --- Cloud enhancement via Band 13 IR ---
    # Only very cold tops (< 235 K — deep convection, thick cirrus anvils) are
    # blended toward bright white.  The enhancement is attenuated near the
    # solar terminator (cos_sza < 0.15) so dark twilight pixels don't
    # accidentally get brightened by a cold cloud top.
    if bt13 is not None:
        bt13_f = np.where(np.isnan(bt13), 320.0, bt13)
        # Band 13 is 2 km; Band 1/3 composite is at 1 km – upsample to match
        if bt13_f.shape != R.shape:
            zy = R.shape[0] / bt13_f.shape[0]
            zx = R.shape[1] / bt13_f.shape[1]
            bt13_f = zoom(bt13_f, (zy, zx), order=1)
        # 235 K → 0 (no enhancement); 190 K → 1 (pure-white deep convection)
        cloud_enhance = np.clip((235.0 - bt13_f) / 45.0, 0.0, 1.0)
        # Fade enhancement to zero near the terminator
        if scan_time is not None:
            lit = np.clip(cos_sza / 0.15, 0.0, 1.0)
            cloud_enhance = cloud_enhance * lit
        strength = 0.50
        R = np.clip(R + cloud_enhance * (1.0 - R) * strength, 0.0, 1.0)
        G = np.clip(G + cloud_enhance * (1.0 - G) * strength, 0.0, 1.0)
        B = np.clip(B + cloud_enhance * (1.0 - B) * strength, 0.0, 1.0)

    rgb = np.dstack([R, G, B])
    fig, ax = _make_figure()
    img_extent = (x[0], x[-1], y[-1], y[0])
    ax.imshow(rgb, origin='upper', extent=img_extent,
              transform=goes_proj, interpolation='none')

    shift_frames('geocolor')
    output_path = os.path.join(OUTPUT_DIR, 'geocolor_00.png')
    _save_png(output_path)
    print(f"  Saved (daytime): {output_path}")


def _render_geocolor_night(s3_client, scan_time):
    """Nighttime GeoColor composite.

    Cloud layer  – derived from Band 13 (10.4 µm clean IR window).
                   Cold temperatures → bright blue-white clouds.
    City lights  – derived from Band 7 (3.9 µm shortwave IR) minus the
                   thermal background estimated from Band 13.  At night,
                   cities, fires, and industrial heat sources emit
                   anomalously in 3.9 µm.
    Background   – fully transparent so the dark basemap shows through.
    """
    # Band 13: brightness temperature (K), same 2 km resolution as Band 7
    bt13, x13, y13, goes_proj = _download_band(s3_client, 13, scan_time)
    if bt13 is None:
        print("  ERROR: Missing Band 13 for nighttime GeoColor. Skipping.")
        return

    # Fill off-earth NaNs with a warm value so they don't look like cloud tops
    bt13 = np.where(np.isnan(bt13), 320.0, bt13)

    # Band 7: 3.9 µm shortwave IR — optional; gracefully absent
    bt7, x7, y7, _ = _download_band(s3_client, 7, scan_time)

    h, w = bt13.shape

    # ------------------------------------------------------------------
    # Cloud layer
    # ------------------------------------------------------------------
    # 275 K → no cloud (opacity 0); 220 K → deep convection (opacity 1)
    cloud_opacity = np.clip((275.0 - bt13) / 55.0, 0.0, 1.0)

    # Clouds rendered as cool blue-white (scattered light / natural night look)
    cloud_R = cloud_opacity * 0.80
    cloud_G = cloud_opacity * 0.88
    cloud_B = cloud_opacity * 1.00

    # ------------------------------------------------------------------
    # City lights layer
    # ------------------------------------------------------------------
    if bt7 is not None:
        bt7 = np.where(np.isnan(bt7), 0.0, bt7)

        # Match Band 7 resolution to Band 13 if needed
        if bt7.shape != (h, w):
            zy = h / bt7.shape[0]
            zx = w / bt7.shape[1]
            bt7 = zoom(bt7, (zy, zx), order=1)

        # At typical surface temperatures Band 7 BT runs ~12 K cooler than
        # Band 13 due to Planck function differences.  Where Band 7 exceeds
        # this offset (i.e. Band7 > Band13 − 12) there is anomalous emission
        # from city lights, fires, or industrial heat.
        city_raw = np.clip((bt7 - (bt13 - 12.0)) / 25.0, 0.0, 1.0)

        # Only paint city lights over clear, warm surface pixels
        surface_clear = (bt13 > 265.0).astype(np.float32)
        city_lights   = city_raw * surface_clear
    else:
        city_lights = np.zeros((h, w), dtype=np.float32)
        print("  WARNING: Band 7 unavailable; city lights layer disabled.")

    # ------------------------------------------------------------------
    # Compose RGBA image
    # ------------------------------------------------------------------
    # Clouds: blue-white  |  City lights: warm yellow-orange
    R = np.clip(cloud_R + city_lights * 1.00, 0.0, 1.0)
    G = np.clip(cloud_G + city_lights * 0.75, 0.0, 1.0)
    B = np.clip(cloud_B + city_lights * 0.10, 0.0, 1.0)

    # Alpha: transparent where there is nothing to show
    A = np.clip(cloud_opacity + city_lights * 2.0, 0.0, 1.0)

    rgba = np.dstack([R, G, B, A]).astype(np.float32)

    fig, ax = _make_figure()
    img_extent = (x13[0], x13[-1], y13[-1], y13[0])
    ax.imshow(rgba, origin='upper', extent=img_extent,
              transform=goes_proj, interpolation='none')

    shift_frames('geocolor')
    output_path = os.path.join(OUTPUT_DIR, 'geocolor_00.png')
    _save_png(output_path)
    print(f"  Saved (nighttime): {output_path}")


# ---------------------------------------------------------------------------
# Experimental precipitation rate product
# ---------------------------------------------------------------------------

def _precip_inputs(s3_client, scan_time):
    """Gather the bands the precipitation estimator runs on.

    Returns (bt13, x, y, sat_proj, bt9, bt15, prev, gap).  bt13 is None when
    the required window channel is unavailable; every other input is optional
    and simply disables the screen it feeds.
    """
    bt13, x, y, sat_proj = _download_band(s3_client, 13, scan_time)
    if bt13 is None:
        return None, None, None, None, None, None, None, 0

    # Band 9 sharpens the deep-convection screen but is not required.
    bt9, *_ = _download_band(s3_client, 9, scan_time)
    if bt9 is None:
        print("  WARNING: Band 9 unavailable; water-vapour screen disabled.")

    # Band 15 gives the split-window thin-cirrus screen.
    bt15, *_ = _download_band(s3_client, 15, scan_time)
    if bt15 is None:
        print("  WARNING: Band 15 unavailable; split-window screen disabled.")

    prev_time, gap = get_previous_scan_time(s3_client, scan_time)
    prev = None
    if prev_time is not None:
        prev, *_ = _download_band(s3_client, 13, prev_time)
        if prev is None:
            gap = 0

    return bt13, x, y, sat_proj, bt9, bt15, prev, gap


def process_precip(s3_client, scan_time):
    """Render the experimental IR-derived precipitation rate (full disk)."""
    print("\n--- Precipitation Rate (experimental, IR-derived) ---")

    bt13, x, y, sat_proj, bt9, bt15, prev, gap = _precip_inputs(s3_client, scan_time)
    if bt13 is None:
        print("  ERROR: Missing Band 13. Skipping precipitation.")
        return

    km = _km_per_pixel(x)
    rate = estimate_precip_rate(bt13, bt_wv=bt9, bt13_prev=prev,
                                prev_minutes=gap, km_per_px=km, x=x, y=y,
                                bt_split=bt15)
    print(f"  Grid {rate.shape[0]}×{rate.shape[1]} @ {km:.1f} km/px — {_precip_stats(rate)}")

    fig, ax = _make_figure()
    ax.imshow(_precip_rgba(rate), origin='upper',
              extent=(x[0], x[-1], y[-1], y[0]), transform=sat_proj,
              interpolation='nearest')

    shift_frames('precip')
    output_path = os.path.join(OUTPUT_DIR, 'precip_00.png')
    _save_png(output_path)
    print(f"  Saved: {output_path}")


def _render_sector_precip(precip_inputs, bounds, product_base):
    """Render the precipitation estimate cropped to one cyclone sector.

    The estimator runs on the cropped arrays rather than on a crop of the
    full-disk result, so the texture window and the cooling trend are computed
    at the sector's own sampling rather than inherited from the disk.
    """
    bt13, x, y, sat_proj, bt9, bt15, prev, gap = precip_inputs
    if bt13 is None:
        return

    bt13s, xs, ys = _crop_band_to_sector(bt13, x, y, sat_proj, bounds)
    bt9s  = _crop_band_to_sector(bt9,  x, y, sat_proj, bounds)[0] if bt9  is not None else None
    bt15s = _crop_band_to_sector(bt15, x, y, sat_proj, bounds)[0] if bt15 is not None else None
    prevs = _crop_band_to_sector(prev, x, y, sat_proj, bounds)[0] if prev is not None else None

    km = _km_per_pixel(xs)
    rate = estimate_precip_rate(bt13s, bt_wv=bt9s, bt13_prev=prevs,
                                prev_minutes=gap, km_per_px=km, x=xs, y=ys,
                                bt_split=bt15s)
    print(f"    Precip: {rate.shape[0]}×{rate.shape[1]} @ {km:.1f} km/px — {_precip_stats(rate)}")

    fig, ax = _make_figure_sector(bounds)
    ax.imshow(_precip_rgba(rate), origin='upper',
              extent=(xs[0], xs[-1], ys[-1], ys[0]), transform=sat_proj,
              interpolation='nearest')

    shift_frames(product_base)
    _save_png(os.path.join(OUTPUT_DIR, f'{product_base}_00.png'), dpi=SECTOR_DPI)
    print(f"    Saved sector precipitation.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _guarded(label, fn, *args):
    """Run a product renderer, reporting rather than raising on failure.

    Returns True on success.  Any open figure is discarded so the next product
    starts from a clean matplotlib state.
    """
    try:
        fn(*args)
        return True
    except Exception as e:
        print(f"\nWARNING: {label} failed: {e}")
        import traceback
        traceback.print_exc()
        plt.close('all')
        return False


def main():
    print("Himawari-9 Satellite Image Processor")
    print("=" * 40)
    print(f"Extent: {EXTENT}")
    print(f"Bucket: s3://{BUCKET}")

    # Anonymous access — Himawari-9 bucket is publicly readable
    s3 = boto3.client(
        's3',
        region_name='us-east-1',
        config=Config(signature_version=UNSIGNED)
    )

    # Find the most recent 10-minute scan with data
    print("\nFinding latest scan...")
    scan_time = get_latest_scan_time(s3)
    if scan_time is None:
        print("ERROR: No Himawari-9 data found in the last 6 hours. Aborting.")
        return

    # Band 3  — Red Visible (0.64 µm)          reflectance [0.0 – 1.0]
    # sun_correct=True multiplies by cos(SZA) to remove the solar-zenith-angle
    # normalisation, restoring physical albedo so the terminator zone fades
    # naturally rather than blowing out to blinding white.
    process_goes_band(s3, 3,  scan_time, 'visible.png',     'gray',          vmin=0.0, vmax=1.0, gamma=0.6, sun_correct=True)

    # Band 13 — Clean IR Longwave (10.4 µm)    brightness temp [K]
    # Custom NWS-style rainbow: cold tops → red/orange, warm surface → dark blue/black
    process_goes_band(s3, 13, scan_time, 'infrared.png',    _ir_colormap(),  vmin=190, vmax=310)

    # Band 9  — Mid-Level Water Vapor (6.9 µm)  brightness temp [K]
    # Custom enhancement: cold/moist → navy/blue, warm/dry → orange/red
    process_goes_band(s3, 9,  scan_time, 'water_vapor.png', _wv_colormap(),  vmin=195, vmax=280)

    # GeoColor — natural colour (day) or IR+city-lights composite (night)
    process_geocolor(s3, scan_time)

    # The remaining products are the ones with outside dependencies — the
    # experimental estimator and the JTWC storm feed — so a failure in either
    # is contained here.  The whole run is committed as one unit, so letting
    # an exception escape would discard the four full-disk images already
    # rendered above and leave the site showing stale frames.
    _guarded("precipitation rate", process_precip, s3, scan_time)

    # Cyclone sectors — high-resolution crops centred on each active storm
    print("\nFetching active storm data for cyclone sectors...")
    storms = fetch_active_storms()
    _guarded("cyclone sectors", process_cyclone_sectors, s3, scan_time, storms)

    # Write a plain-text timestamp so the website can show freshness
    ts_path = os.path.join(OUTPUT_DIR, 'last_updated.txt')
    with open(ts_path, 'w') as f:
        f.write(datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC'))
    print(f"\nTimestamp written: {ts_path}")
    print("\nDone!")


if __name__ == '__main__':
    main()
