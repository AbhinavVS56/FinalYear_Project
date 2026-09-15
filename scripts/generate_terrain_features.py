import rasterio
import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter
from pathlib import Path

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEM_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "dem"
    / "idukki_dem_utm43n.tif"
)

OUTPUT_DIR = PROJECT_ROOT / "datasets" / "processed" / "terrain_features"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DEM
# ============================================================

print("Loading Idukki DEM...")

with rasterio.open(DEM_PATH) as src:

    elevation = src.read(1).astype(np.float32)
    profile = src.profile.copy()

    pixel_x = src.transform.a
    pixel_y = abs(src.transform.e)

    nodata = src.nodata

print("DEM loaded.")
print("Shape:", elevation.shape)
print("Pixel size:", pixel_x, "x", pixel_y, "meters")


# ============================================================
# HANDLE NODATA
# ============================================================

if nodata is not None:
    valid_mask = elevation != nodata
else:
    valid_mask = np.isfinite(elevation)

elevation = np.where(valid_mask, elevation, np.nan)


# ============================================================
# 1. ELEVATION
# ============================================================

print("\nGenerating Elevation...")

elevation_output = OUTPUT_DIR / "idukki_elevation.tif"

elevation_save = np.where(
    valid_mask,
    elevation,
    -9999
).astype(np.float32)

elevation_profile = profile.copy()
elevation_profile.update(
    dtype="float32",
    count=1,
    nodata=-9999,
    compress="lzw"
)

with rasterio.open(elevation_output, "w", **elevation_profile) as dst:
    dst.write(elevation_save, 1)

print("Elevation saved.")


# ============================================================
# FILL NAN FOR DERIVATIVES
# ============================================================

# Temporary filled DEM for calculations
filled = np.nan_to_num(
    elevation,
    nan=np.nanmedian(elevation)
).astype(np.float32)


# ============================================================
# 2. SLOPE
# ============================================================

print("\nGenerating Slope...")

dy, dx = np.gradient(
    filled,
    pixel_y,
    pixel_x
)

slope = np.degrees(
    np.arctan(
        np.sqrt(dx**2 + dy**2)
    )
).astype(np.float32)

slope = np.where(valid_mask, slope, -9999)

slope_output = OUTPUT_DIR / "idukki_slope.tif"

with rasterio.open(
    slope_output,
    "w",
    **elevation_profile
) as dst:
    dst.write(slope, 1)

print("Slope saved.")
print(
    "Slope range:",
    np.nanmin(np.where(valid_mask, slope, np.nan)),
    "to",
    np.nanmax(np.where(valid_mask, slope, np.nan))
)


# ============================================================
# 3. ASPECT
# ============================================================

print("\nGenerating Aspect...")

aspect = np.degrees(
    np.arctan2(-dx, dy)
)

aspect = (aspect + 360) % 360

aspect = aspect.astype(np.float32)

aspect = np.where(valid_mask, aspect, -9999)

aspect_output = OUTPUT_DIR / "idukki_aspect.tif"

with rasterio.open(
    aspect_output,
    "w",
    **elevation_profile
) as dst:
    dst.write(aspect, 1)

print("Aspect saved.")


# ============================================================
# 4. CURVATURE
# ============================================================

print("\nGenerating Curvature...")

# Small Gaussian smoothing reduces tiny DEM noise
smooth = gaussian_filter(
    filled,
    sigma=1
)

dy_s, dx_s = np.gradient(
    smooth,
    pixel_y,
    pixel_x
)

dyy, _ = np.gradient(
    dy_s,
    pixel_y,
    pixel_x
)

_, dxx = np.gradient(
    dx_s,
    pixel_y,
    pixel_x
)

curvature = (
    dxx + dyy
).astype(np.float32)

curvature = np.where(
    valid_mask,
    curvature,
    -9999
)

curvature_output = OUTPUT_DIR / "idukki_curvature.tif"

with rasterio.open(
    curvature_output,
    "w",
    **elevation_profile
) as dst:
    dst.write(curvature, 1)

print("Curvature saved.")


# ============================================================
# 5. TRI
# ============================================================

print("\nGenerating TRI...")

# 3x3 neighborhood mean
mean = uniform_filter(
    filled,
    size=3,
    mode="nearest"
)

# Mean squared difference between center and neighbors
squared_difference = (
    filled - mean
) ** 2

tri = np.sqrt(
    squared_difference
).astype(np.float32)

tri = np.where(
    valid_mask,
    tri,
    -9999
)

tri_output = OUTPUT_DIR / "idukki_tri.tif"

with rasterio.open(
    tri_output,
    "w",
    **elevation_profile
) as dst:
    dst.write(tri, 1)

print("TRI saved.")


# ============================================================
# 6. HILLSHADE
# ============================================================

print("\nGenerating Hillshade...")

azimuth = 315
altitude = 45

azimuth_rad = np.radians(azimuth)
altitude_rad = np.radians(altitude)

slope_rad = np.arctan(
    np.sqrt(dx**2 + dy**2)
)

aspect_rad = np.arctan2(
    -dx,
    dy
)

hillshade = (
    np.sin(altitude_rad) * np.sin(slope_rad)
    +
    np.cos(altitude_rad)
    * np.cos(slope_rad)
    * np.cos(azimuth_rad - aspect_rad)
)

hillshade = (
    hillshade * 255
).astype(np.float32)

hillshade = np.clip(
    hillshade,
    0,
    255
)

hillshade = np.where(
    valid_mask,
    hillshade,
    -9999
)

hillshade_output = OUTPUT_DIR / "idukki_hillshade.tif"

with rasterio.open(
    hillshade_output,
    "w",
    **elevation_profile
) as dst:
    dst.write(hillshade, 1)

print("Hillshade saved.")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n======================================")
print("TERRAIN FEATURE GENERATION COMPLETE")
print("======================================")

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nGenerated files:")

for file in OUTPUT_DIR.glob("*.tif"):
    print(" -", file.name)

print("\nAll terrain layers use:")
print("CRS: EPSG:32643")
print("Resolution:", pixel_x, "x", pixel_y, "meters")