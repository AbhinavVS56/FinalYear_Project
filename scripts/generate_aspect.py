import rasterio
import numpy as np
import matplotlib.pyplot as plt

dem_path = r"E:\Projects\Final Year Project\datasets\processed\dem\cop30\rasters_COP30\wayanad_cop30_dem.tif"

with rasterio.open(dem_path) as dem:
    elevation = dem.read(1)
    gradient_y, gradient_x = np.gradient(elevation)
    aspect = np.arctan2(-gradient_y, gradient_x)
    aspect = np.degrees(aspect)
    aspect = (450 - aspect) % 360
    
    plt.figure(figsize=(10, 8))
    plt.imshow(aspect, cmap="hsv")
    plt.colorbar(label="Aspect (Degrees)")
    plt.title("Aspect Map")
    plt.xlabel("Columns")
    plt.ylabel("Rows")
    plt.show()