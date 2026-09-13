import rasterio
import numpy as np

dem_path = r"C:\Users\user\Downloads\Final Year Project\datasets\raw\dem\cop30\idukki_cop30_dem.tif"

with rasterio.open(dem_path) as dem:

    print("========== DEM INFORMATION ==========")

    print("CRS:", dem.crs)
    print("Width:", dem.width)
    print("Height:", dem.height)

    print("Resolution:", dem.res)

    print("Bounds:")
    print(dem.bounds)

    print("Data type:", dem.dtypes[0])
    print("NoData:", dem.nodata)

    elevation = dem.read(1)

    if dem.nodata is not None:
        elevation = elevation[elevation != dem.nodata]

    print("\nElevation Statistics:")
    print("Minimum:", np.min(elevation))
    print("Maximum:", np.max(elevation))
    print("Mean:", np.mean(elevation))
    print("Std:", np.std(elevation))