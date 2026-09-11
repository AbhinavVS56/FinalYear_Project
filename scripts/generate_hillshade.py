import rasterio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

dem_path = BASE_DIR / "datasets" / "processed" / "dem" / "cop30" / "rasters_COP30" / "wayanad_cop30_dem.tif"

output_path = BASE_DIR / "datasets" / "processed" / "hillshade" / "wayanad_hillshade.tif"

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


slope = np.arctan(
    np.sqrt(
        gradient_x ** 2 +
        gradient_y ** 2
    )
)

aspect = np.arctan2(
    -gradient_y,
    gradient_x
)

# Direction from which the light comes.
# 315° = Northwest

azimuth = np.radians(315)

# Height of the light source above the horizon.
# 45° = light is halfway between horizon
# and directly overhead.

altitude = np.radians(45)

# Zenith is measured from straight overhead.
# Therefore:
#
# Zenith = 90° - altitude

zenith = np.pi / 2 - altitude

print("\nCalculating hillshade...")

hillshade = (
    np.cos(zenith) * np.cos(slope)
    +
    np.sin(zenith)
    * np.sin(slope)
    * np.cos(azimuth - aspect)
)

hillshade = 255 * hillshade

hillshade = np.clip(
    hillshade,
    0,
    255
)

print("\nHillshade Statistics:")

print(f"Minimum: {np.nanmin(hillshade):.2f}")
print(f"Maximum: {np.nanmax(hillshade):.2f}")
print(f"Mean:    {np.nanmean(hillshade):.2f}")
print(f"Std Dev: {np.nanstd(hillshade):.2f}")


print("\nSaving hillshade raster...")

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
        hillshade.astype("float32"),
        1
    )

print("Hillshade raster saved to:")
print(output_path)

print("\nPreparing visualization...")

plt.figure(figsize=(10, 8))

plt.imshow(
    hillshade,
    cmap="gray",
    vmin=0,
    vmax=255
)

plt.colorbar(
    label="Brightness"
)

plt.title("Wayanad Hillshade Map")

plt.xlabel("Columns")
plt.ylabel("Rows")

plt.show()