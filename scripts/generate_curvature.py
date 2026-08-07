import rasterio
import matplotlib.pyplot as plt
import numpy as np

dem_path=r"E:\Projects\Final Year Project\datasets\processed\dem\cop30\rasters_COP30\wayanad_cop30_dem.tif"

with rasterio.open(dem_path) as dem:
    elevation = dem.read(1)
dx, dy = np.gradient(elevation)
dxx, _ = np.gradient(dx)
_, dyy = np.gradient(dy)
curvature = dxx + dyy

plt.figure(figsize=(10, 8))
plt.imshow(curvature, cmap="RdBu")
plt.colorbar(label="Curvature")
plt.title("Curvature Map")
plt.xlabel("Columns")
plt.ylabel("Rows")
plt.show()