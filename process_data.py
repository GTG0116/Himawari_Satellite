import boto3
import shutil
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
from scipy.ndimage import zoom
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
# Our output figure is 12 in × 300 DPI = 3 600 px wide, so 2 750 px input
# is already more than enough and keeps each band under ~30 MB of RAM.
# Band 3 (0.5 km, 11 000×11 000) is downsampled 4× to ~2 750×2 750;
# 2 km bands (5 500×5 500) are halved to ~2 750×2 750.
MAX_BAND_PX = 2750

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


def _download_band(s3_client, band_num, scan_time):
    """Download all HSD segments for a Himawari-9 AHI band and load with satpy.

    Results are cached for the lifetime of the process so that bands shared
    between products (Band 3 for visible + GeoColor, Band 13 for IR + GeoColor)
    are only downloaded and decoded once.

    Returns (data_array, x_metres, y_metres, sat_proj).
    data_array contains calibrated float32 values (reflectance or BT) with NaN
    for off-Earth pixels.  Returns (None, None, None, None) on any failure.
    """
    cache_key = (band_num, scan_time)
    if cache_key in _band_cache:
        print(f'  Band {band_num}: using cached data')
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
        area    = ds.attrs['area']
        left, bottom, right, top = area.area_extent
        n_rows, n_cols = arr.shape

        # Downsample high-resolution bands to stay within MAX_BAND_PX.
        # Band 3 (0.5 km) is 11 000×11 000 ≈ 484 MB; downsampled 4× → ~30 MB.
        # 2 km bands (5 500×5 500) are halved → ~30 MB.
        # Our output figure is 3 600 px wide, so 2 750 px input is plenty.
        if max(n_rows, n_cols) > MAX_BAND_PX:
            scale = MAX_BAND_PX / max(n_rows, n_cols)
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
    plt.savefig(output_path, dpi=300, transparent=True)
    plt.close()
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
    plt.savefig(output_path, dpi=300, transparent=True)
    plt.close()
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
    plt.savefig(output_path, dpi=300, transparent=True)
    plt.close()
    print(f"  Saved (nighttime): {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

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

    # Write a plain-text timestamp so the website can show freshness
    ts_path = os.path.join(OUTPUT_DIR, 'last_updated.txt')
    with open(ts_path, 'w') as f:
        f.write(datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC'))
    print(f"\nTimestamp written: {ts_path}")
    print("\nDone!")


if __name__ == '__main__':
    main()
