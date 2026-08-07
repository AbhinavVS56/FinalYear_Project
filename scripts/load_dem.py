import rasterio
import matplotlib.pyplot as plt
import numpy as np

dem_path=r"E:\Projects\Final Year Project\datasets\processed\dem\cop30\rasters_COP30\wayanad_cop30_dem.tif"

with rasterio.open(dem_path) as dem:
    elevation=dem.read(1)

    gradient_y,gradient_x=np.gradient(elevation)
    slope=np.sqrt(gradient_x**2+gradient_y**2)

    plt.figure(figsize=(10, 8))
    plt.imshow(elevation, cmap="terrain")
    plt.colorbar(label="Elevation (meters)")
    plt.title("Wayanad Copernicus DEM")
    plt.xlabel("Columns")
    plt.ylabel("Rows")
    plt.show()

    plt.figure(figsize=(10, 8))
    plt.imshow(slope, cmap="gray")
    plt.colorbar(label="Slope")
    plt.title("Slope Map")
    plt.xlabel("Columns")
    plt.ylabel("Rows")
    plt.show()