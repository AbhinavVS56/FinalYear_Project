import rasterio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

dem_path = BASE_DIR / "datasets" / "processed" / "dem" / "cop30" / "rasters_COP30" / "wayanad_cop30_dem.tif"

slope_output_path = BASE_DIR / "datasets" / "processed" / "slope" / "wayanad_slope.tif"


print("Loading DEM...")

with rasterio.open(dem_path) as dem:
    elevation = dem.read(1).astype("float64")
    profile = dem.profile
    bounds = dem.bounds

print("DEM loaded.")

print(f"DEM shape: {elevation.shape}")

print("\nElevation Statistics:")

print(f"Minimum Elevation: {np.nanmin(elevation):.2f} m")
print(f"Maximum Elevation: {np.nanmax(elevation):.2f} m")
print(f"Mean Elevation:    {np.nanmean(elevation):.2f} m")
print(f"Std Dev:           {np.nanstd(elevation):.2f} m")


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

print(
    f"North-South: {pixel_height_m:.2f} meters"
)

print(
    f"East-West:   {pixel_width_m:.2f} meters"
)

print("\nCalculating elevation gradients...")

gradient_y, gradient_x = np.gradient(
    elevation,
    pixel_height_m,
    pixel_width_m
)

# Gradient magnitude represents
# how quickly elevation changes
# with horizontal distance.

gradient_magnitude = np.sqrt(
    gradient_x ** 2 +
    gradient_y ** 2
)


# Convert gradient to slope angle in radians

slope_radians = np.arctan(
    gradient_magnitude
)


# Convert radians to degrees

slope = np.degrees(
    slope_radians
)

print("\nSlope Statistics:")

print(
    f"Minimum: {np.nanmin(slope):.2f} degrees"
)

print(
    f"Maximum: {np.nanmax(slope):.2f} degrees"
)

print(
    f"Mean:    {np.nanmean(slope):.2f} degrees"
)

print(
    f"Std Dev: {np.nanstd(slope):.2f} degrees"
)

print("\nSaving slope raster...")

profile.update(
    dtype="float32",
    count=1,
    compress="lzw"
)

with rasterio.open(
    slope_output_path,
    "w",
    **profile
) as dst:

    dst.write(
        slope.astype("float32"),
        1
    )

print("Slope raster saved to:")

print(slope_output_path)

plt.figure(figsize=(10, 8))

plt.imshow(
    elevation,
    cmap="terrain"
)

plt.colorbar(
    label="Elevation (meters)"
)

plt.title(
    "Wayanad Copernicus DEM"
)

plt.xlabel("Columns")
plt.ylabel("Rows")

plt.show()


plt.figure(figsize=(10, 8))

plt.imshow(
    slope,
    cmap="gray",
    vmin=0,
    vmax=np.percentile(slope, 98)
)

plt.colorbar(
    label="Slope (Degrees)"
)

plt.title(
    "Wayanad Slope Map"
)

plt.xlabel("Columns")
plt.ylabel("Rows")

plt.show()