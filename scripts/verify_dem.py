import rasterio
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

dem_path = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "dem"
    / "idukki_dem_utm43n.tif"
)

with rasterio.open(dem_path) as dem:

    elevation = dem.read(1)

    print("========== FINAL DEM ==========")
    print("CRS:", dem.crs)
    print("Width:", dem.width)
    print("Height:", dem.height)
    print("Resolution:", dem.res)
    print("Bounds:", dem.bounds)
    print("Data type:", dem.dtypes[0])
    print("NoData:", dem.nodata)

    print("\nElevation Statistics:")
    print("Minimum:", np.min(elevation))
    print("Maximum:", np.max(elevation))
    print("Mean:", np.mean(elevation))
    print("Std:", np.std(elevation))