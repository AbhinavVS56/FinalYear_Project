import rasterio
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


dem_path = r"E:\Projects\Final Year Project\datasets\processed\dem\cop30\rasters_COP30\wayanad_cop30_dem.tif"

output_path = r"E:\Projects\Final Year Project\datasets\processed\curvature\wayanad_curvature.tif"

with rasterio.open(dem_path) as dem:

    elevation = dem.read(1)

    profile = dem.profile

    pixel_width_deg = dem.res[0]
    pixel_height_deg = dem.res[1]

    bounds = dem.bounds

mean_latitude = (bounds.top + bounds.bottom) / 2
meters_per_degree_lat = 111132.0
meters_per_degree_lon = (
    111320.0 * np.cos(np.radians(mean_latitude))
)
pixel_height_m = pixel_height_deg * meters_per_degree_lat
pixel_width_m = pixel_width_deg * meters_per_degree_lon
print("Pixel spacing:")
print(f"North-South: {pixel_height_m:.2f} meters")
print(f"East-West:   {pixel_width_m:.2f} meters")

elevation_smooth = gaussian_filter(
    elevation,
    sigma=1
)

dy, dx = np.gradient(
    elevation_smooth,
    pixel_height_m,
    pixel_width_m
)

dyy, _ = np.gradient(
    dy,
    pixel_height_m,
    pixel_width_m
)

_, dxx = np.gradient(
    dx,
    pixel_height_m,
    pixel_width_m
)

curvature = dxx + dyy


print("\nCurvature Statistics:")

print(f"Minimum: {np.nanmin(curvature):.8f}")
print(f"Maximum: {np.nanmax(curvature):.8f}")
print(f"Mean:    {np.nanmean(curvature):.8f}")
print(f"Std Dev: {np.nanstd(curvature):.8f}")


profile.update(
    dtype="float32",
    count=1,
    compress="lzw"
)

with rasterio.open(output_path, "w", **profile) as dst:

    dst.write(
        curvature.astype("float32"),
        1
    )

print("\nCurvature raster saved to:")
print(output_path)

lower = np.percentile(curvature, 2)
upper = np.percentile(curvature, 98)

limit = max(abs(lower), abs(upper))

plt.figure(figsize=(10, 8))

plt.imshow(
    curvature,
    cmap="RdBu_r",
    vmin=-limit,
    vmax=limit
)

plt.colorbar(
    label="Curvature"
)

plt.title("Wayanad Curvature Map")

plt.xlabel("Columns")
plt.ylabel("Rows")

plt.show()