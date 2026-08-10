import rasterio
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter


dem_path = r"E:\Projects\Final Year Project\datasets\processed\dem\cop30\rasters_COP30\wayanad_cop30_dem.tif"

output_path = r"E:\Projects\Final Year Project\datasets\processed\tri\wayanad_tri.tif"

print("Loading DEM...")

with rasterio.open(dem_path) as dem:

    elevation = dem.read(1).astype("float64")

    profile = dem.profile

    nodata = dem.nodata

print("DEM loaded.")
print(f"DEM shape: {elevation.shape}")

print("Calculating neighborhood statistics...")

# 3x3 neighborhood
# The center pixel will be removed later.

window_size = 3

# Sum of elevations in each 3x3 neighborhood
sum_elevation = uniform_filter(
    elevation,
    size=window_size,
    mode="nearest"
) * 9

# Sum of squared elevations in each 3x3 neighborhood
sum_squared = uniform_filter(
    elevation ** 2,
    size=window_size,
    mode="nearest"
) * 9

# Current pixel elevation
center = elevation

# Sum of the 8 neighboring pixels
neighbor_sum = sum_elevation - center

# Sum of squares of the 8 neighboring pixels
neighbor_squared_sum = sum_squared - center ** 2

# Average neighboring elevation
neighbor_mean = neighbor_sum / 8.0

# Average squared neighboring elevation
neighbor_squared_mean = neighbor_squared_sum / 8.0

# Mean squared difference between
# the center and its 8 neighbors:
#
# mean((neighbor - center)^2)
#
# = mean(neighbor^2)
#   - 2 * center * mean(neighbor)
#   + center^2

mean_squared_difference = (
    neighbor_squared_mean
    - (2 * center * neighbor_mean)
    + center ** 2
)

mean_squared_difference = np.maximum(
    mean_squared_difference,
    0
)

tri = np.sqrt(
    mean_squared_difference
)


print("\nTRI Statistics:")

print(f"Minimum: {np.nanmin(tri):.4f}")
print(f"Maximum: {np.nanmax(tri):.4f}")
print(f"Mean:    {np.nanmean(tri):.4f}")
print(f"Std Dev: {np.nanstd(tri):.4f}")


print("\nSaving TRI raster...")

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
        tri.astype("float32"),
        1
    )

print("TRI raster saved to:")
print(output_path)

print("\nPreparing visualization...")

lower = np.percentile(tri, 2)
upper = np.percentile(tri, 98)

plt.figure(figsize=(10, 8))

plt.imshow(
    tri,
    cmap="terrain",
    vmin=lower,
    vmax=upper
)

plt.colorbar(
    label="Terrain Ruggedness Index (m)"
)

plt.title("Wayanad Terrain Ruggedness Index")

plt.xlabel("Columns")
plt.ylabel("Rows")

plt.show()