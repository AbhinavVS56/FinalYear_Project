import rasterio
import matplotlib.pyplot as plt
import numpy as np


dem_path = r"E:\Projects\Final Year Project\datasets\processed\dem\cop30\rasters_COP30\wayanad_cop30_dem.tif"

with rasterio.open(dem_path) as dem:
    elevation = dem.read(1)

gradient_y, gradient_x = np.gradient(elevation)
slope = np.pi / 2 - np.arctan(np.sqrt(gradient_x**2 + gradient_y**2))
aspect = np.arctan2(-gradient_y, gradient_x)
azimuth = np.radians(315)     
altitude = np.radians(45)     
zenith = np.pi / 2 - altitude
hillshade = (
    np.cos(zenith) * np.cos(slope)
    + np.sin(zenith) * np.sin(slope) * np.cos(azimuth - aspect)
)
hillshade = 255 * hillshade
hillshade = np.clip(hillshade, 0, 255)

plt.figure(figsize=(10,8))
plt.imshow(hillshade, cmap="gray")
plt.title("Hillshade Map")
plt.xlabel("Columns")
plt.ylabel("Rows")
plt.colorbar(label="Brightness")
plt.show()