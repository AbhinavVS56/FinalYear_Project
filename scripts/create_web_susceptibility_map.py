import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling, transform_bounds
from PIL import Image

INPUT = Path('datasets/processed/susceptibility/idukki_static_susceptibility.tif')
OUT_DIR = Path('frontend/map_data')
OUT_DIR.mkdir(parents=True, exist_ok=True)
PNG = OUT_DIR / 'idukki_static_susceptibility.png'
BOUNDS = OUT_DIR / 'idukki_map_bounds.json'

# Web display palette: low -> moderate -> high -> very high.
# These are visualization bins, not calibrated warning thresholds.
STOPS = np.array([0.0, 0.25, 0.50, 0.75, 1.0], dtype=np.float32)
COLORS = np.array([
    [35, 94, 63],
    [170, 157, 53],
    [205, 104, 54],
    [170, 55, 62],
    [125, 35, 45],
], dtype=np.float32)

with rasterio.open(INPUT) as src:
    if src.crs is None:
        raise RuntimeError('Input raster has no CRS.')

    transform, width, height = calculate_default_transform(
        src.crs, 'EPSG:4326', src.width, src.height, *src.bounds
    )

    data = np.full((height, width), np.nan, dtype=np.float32)

    reproject(
        source=rasterio.band(src, 1),
        destination=data,
        src_transform=src.transform,
        src_crs=src.crs,
        src_nodata=src.nodata,
        dst_transform=transform,
        dst_crs='EPSG:4326',
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )

    valid = np.isfinite(data)
    if not valid.any():
        raise RuntimeError('No valid susceptibility pixels found.')

    data = np.clip(data, 0.0, 1.0)

    # Interpolate between palette stops.
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    for c in range(3):
        rgb[..., c] = np.interp(data, STOPS, COLORS[:, c]).astype(np.uint8)

    # Transparent NoData outside Idukki.
    alpha = np.where(valid, 235, 0).astype(np.uint8)
    rgba = np.dstack([rgb, alpha])

    Image.fromarray(rgba, 'RGBA').save(PNG, optimize=True)

    left = transform.c
    top = transform.f
    right = left + transform.a * width
    bottom = top + transform.e * height

    bounds = {
        'south': float(min(bottom, top)),
        'west': float(min(left, right)),
        'north': float(max(bottom, top)),
        'east': float(max(left, right)),
        'crs': 'EPSG:4326'
    }

    BOUNDS.write_text(json.dumps(bounds, indent=2))

print('Web susceptibility map created.')
print(f'PNG: {PNG}')
print(f'Bounds: {json.dumps(bounds)}')
print('Display classes: 0-25 Low, 25-50 Moderate, 50-75 High, 75-100 Very High.')
print('These classes are visualization bins, not calibrated warning thresholds.')
