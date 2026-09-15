import rasterio
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter

DEM_PATH = Path("../datasets/processed/dem/idukki_dem_utm43n.tif")
OUTPUT_DIR = Path("../datasets/processed/terrain_features")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_raster(path, data, profile):
    profile = profile.copy()
    profile.update(
        dtype="float32",
        count=1,
        nodata=np.nan,
        compress="lzw"
    )

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype("float32"), 1)


print("Loading DEM...")

with rasterio.open(DEM_PATH) as src:
    dem = src.read(1).astype(np.float32)
    profile = src.profile.copy()
    transform = src.transform

print("DEM loaded.")
print("Shape:", dem.shape)
print("CRS:", profile["crs"])
print("Resolution:", abs(transform.a))

# Valid DEM cells
valid = np.isfinite(dem) & (dem > 0)

print("Valid pixels:", np.sum(valid))

# ---------------------------------------------------------
# IMPORTANT:
# Derivatives are calculated ONLY where a complete
# 3x3 neighbourhood is valid.
# No artificial elevation filling is used.
# ---------------------------------------------------------

# Valid 3x3 neighbourhood
valid_neighbourhood = valid.copy()

for dy in (-1, 0, 1):
    for dx in (-1, 0, 1):
        shifted = np.zeros_like(valid)
        y1 = max(0, dy)
        y2 = min(valid.shape[0], valid.shape[0] + dy)
        x1 = max(0, dx)
        x2 = min(valid.shape[1], valid.shape[1] + dx)

        sy1 = max(0, -dy)
        sy2 = min(valid.shape[0], valid.shape[0] - dy)
        sx1 = max(0, -dx)
        sx2 = min(valid.shape[1], valid.shape[1] - dx)

        shifted[y1:y2, x1:x2] = valid[sy1:sy2, sx1:sx2]

        valid_neighbourhood &= shifted

# ---------------------------------------------------------
# Elevation
# ---------------------------------------------------------

print("Generating elevation...")

elevation = np.where(valid, dem, np.nan).astype(np.float32)

save_raster(
    OUTPUT_DIR / "idukki_elevation.tif",
    elevation,
    profile
)

# ---------------------------------------------------------
# Slope + Aspect
# ---------------------------------------------------------

print("Generating slope and aspect...")

# Temporary array only for numerical differentiation.
# Boundary cells are invalidated afterward.
work = dem.astype(np.float64)

# Fill invalid cells only temporarily so np.gradient can run.
# These cells themselves AND their immediate neighbours
# are later removed using valid_neighbourhood.
work[~valid] = np.nanmean(dem[valid])

dy, dx = np.gradient(
    work,
    abs(transform.e),
    abs(transform.a)
)

slope = np.degrees(
    np.arctan(np.sqrt(dx ** 2 + dy ** 2))
)

aspect = np.degrees(
    np.arctan2(-dx, dy)
)

aspect = (aspect + 360) % 360

slope[~valid_neighbourhood] = np.nan
aspect[~valid_neighbourhood] = np.nan

save_raster(
    OUTPUT_DIR / "idukki_slope.tif",
    slope,
    profile
)

save_raster(
    OUTPUT_DIR / "idukki_aspect.tif",
    aspect,
    profile
)

# ---------------------------------------------------------
# Curvature
# ---------------------------------------------------------

print("Generating curvature...")

smooth = gaussian_filter(work, sigma=1)

gy, gx = np.gradient(
    smooth,
    abs(transform.e),
    abs(transform.a)
)

gyy, _ = np.gradient(
    gy,
    abs(transform.e),
    abs(transform.a)
)

_, gxx = np.gradient(
    gx,
    abs(transform.e),
    abs(transform.a)
)

curvature = gxx + gyy

curvature[~valid_neighbourhood] = np.nan

save_raster(
    OUTPUT_DIR / "idukki_curvature.tif",
    curvature,
    profile
)

# ---------------------------------------------------------
# TRI
# ---------------------------------------------------------

print("Generating TRI...")

tri_sum = np.zeros_like(dem, dtype=np.float64)
tri_count = np.zeros_like(dem, dtype=np.uint8)

center = dem.astype(np.float64)

for dy in (-1, 0, 1):
    for dx in (-1, 0, 1):

        if dy == 0 and dx == 0:
            continue

        shifted = np.full_like(center, np.nan)

        y1 = max(0, dy)
        y2 = min(center.shape[0], center.shape[0] + dy)
        x1 = max(0, dx)
        x2 = min(center.shape[1], center.shape[1] + dx)

        sy1 = max(0, -dy)
        sy2 = min(center.shape[0], center.shape[0] - dy)
        sx1 = max(0, -dx)
        sx2 = min(center.shape[1], center.shape[1] - dx)

        shifted[y1:y2, x1:x2] = center[sy1:sy2, sx1:sx2]

        valid_pair = valid & np.isfinite(shifted)

        diff = shifted - center

        tri_sum[valid_pair] += diff[valid_pair] ** 2
        tri_count[valid_pair] += 1

tri = np.sqrt(
    tri_sum / np.maximum(tri_count, 1)
)

# Require complete 8-neighbourhood
tri[~valid_neighbourhood] = np.nan

save_raster(
    OUTPUT_DIR / "idukki_tri.tif",
    tri,
    profile
)

# ---------------------------------------------------------
# Hillshade
# ---------------------------------------------------------

print("Generating hillshade...")

azimuth = np.radians(315)
altitude = np.radians(45)

slope_rad = np.radians(slope)
aspect_rad = np.radians(aspect)

hillshade = (
    np.sin(altitude) * np.cos(slope_rad)
    +
    np.cos(altitude)
    * np.sin(slope_rad)
    * np.cos(azimuth - aspect_rad)
)

hillshade = 255 * hillshade

hillshade[~valid_neighbourhood] = np.nan

save_raster(
    OUTPUT_DIR / "idukki_hillshade.tif",
    hillshade,
    profile
)

print("\n" + "=" * 60)
print("TERRAIN FEATURE GENERATION COMPLETE")
print("=" * 60)