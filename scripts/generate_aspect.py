import rasterio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

dem_path = BASE_DIR / "datasets" / "processed" / "dem" / "cop30" / "rasters_COP30" / "wayanad_cop30_dem.tif"

output_path = BASE_DIR / "datasets" / "processed" / "aspect" / "wayanad_aspect.tif"


print("Loading DEM...")

with rasterio.open(dem_path) as dem:

    elevation = dem.read(1).astype("float64")

    profile = dem.profile

    bounds = dem.bounds

print("DEM loaded.")
print(f"DEM shape: {elevation.shape}")


pixel_width_deg = profile["transform"].a
pixel_height_deg = abs(profile["transform"].e)

mean_latitude = (
    bounds.top + bounds.bottom
) / 2

meters_per_degree_lat = 111132.0

meters_per_degree_lon = (
    111320.0 *
    np.cos(np.radians(mean_latitude))
)

pixel_height_m = (
    pixel_height_deg *
    meters_per_degree_lat
)

pixel_width_m = (
    pixel_width_deg *
    meters_per_degree_lon
)

print("\nPixel spacing:")
print(f"North-South: {pixel_height_m:.2f} meters")
print(f"East-West:   {pixel_width_m:.2f} meters")


print("\nCalculating elevation gradients...")

gradient_y, gradient_x = np.gradient(
    elevation,
    pixel_height_m,
    pixel_width_m
)


print("Calculating aspect...")

# Because raster rows increase downward,
# gradient_y points toward the south.
#
# Therefore -gradient_y represents
# the northward elevation gradient.

aspect = np.arctan2(
    -gradient_y,
    gradient_x
)

aspect = np.degrees(aspect)

# Convert mathematical angle to
# compass direction:
#
# 0°   = North
# 90°  = East
# 180° = South
# 270° = West

aspect = (450 - aspect) % 360

print("\nAspect Statistics:")

print(f"Minimum: {np.nanmin(aspect):.2f} degrees")
print(f"Maximum: {np.nanmax(aspect):.2f} degrees")
print(f"Mean:    {np.nanmean(aspect):.2f} degrees")
print(f"Std Dev: {np.nanstd(aspect):.2f} degrees")


print("\nSaving aspect raster...")

profile.update(
    dtype="float32",
    count=1,
    compress="lzw"
)

with rasterio.open(
    output_path,
    "w",
    **profile
) as dst:

    dst.write(
        aspect.astype("float32"),
        1
    )

print("Aspect raster saved to:")
print(output_path)

print("\nPreparing visualization...")

plt.figure(figsize=(10, 8))

plt.imshow(
    aspect,
    cmap="hsv",
    vmin=0,
    vmax=360
)

plt.colorbar(
    label="Aspect (Degrees)"
)

plt.title("Wayanad Aspect Map")

plt.xlabel("Columns")
plt.ylabel("Rows")

plt.show()