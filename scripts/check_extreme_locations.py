import rasterio
import numpy as np
import geopandas as gpd
from pathlib import Path
from rasterio.features import geometry_mask
from scipy.ndimage import distance_transform_edt

terrain_dir = Path("../datasets/processed/terrain_features")
dem_path = Path("../datasets/processed/dem/idukki_dem_utm43n.tif")
boundary_path = Path("../datasets/raw/boundaries/idukki_boundary.gpkg")


def check_feature(name, threshold):
    path = terrain_dir / f"idukki_{name}.tif"

    with rasterio.open(path) as src:
        data = src.read(1)
        transform = src.transform
        crs = src.crs

        rows, cols = np.where(
            np.isfinite(data) & (data > threshold)
        )

        print("\n" + "=" * 70)
        print(f"{name.upper()} > {threshold}")
        print("=" * 70)

        print(f"Number of pixels: {len(rows)}")

        for row, col in zip(rows[:20], cols[:20]):
            x, y = rasterio.transform.xy(
                transform, row, col, offset="center"
            )

            print(
                f"row={row:4d}, col={col:4d}, "
                f"x={x:.2f}, y={y:.2f}, "
                f"value={data[row, col]:.4f}"
            )


# Load DEM
with rasterio.open(dem_path) as src:
    dem = src.read(1)
    transform = src.transform
    crs = src.crs

# Load boundary
boundary = gpd.read_file(boundary_path)
boundary = boundary.to_crs(crs)

# Create boundary mask
inside = geometry_mask(
    boundary.geometry,
    out_shape=dem.shape,
    transform=transform,
    invert=True
)

# Distance from every inside pixel to boundary
distance = distance_transform_edt(inside) * abs(transform.a)

print("=" * 70)
print("EXTREME TERRAIN LOCATION CHECK")
print("=" * 70)

print(f"DEM CRS: {crs}")
print(f"Pixel size: {abs(transform.a):.2f} m")

check_feature("tri", 100)
check_feature("curvature", 0.05)
check_feature("slope", 60)

print("\n" + "=" * 70)
print("BOUNDARY DISTANCE OF TOP TRI VALUES")
print("=" * 70)

tri_path = terrain_dir / "idukki_tri.tif"

with rasterio.open(tri_path) as src:
    tri = src.read(1)

    valid = np.isfinite(tri)

    rows, cols = np.where(valid)

    values = tri[rows, cols]

    # Top 20 TRI values
    indices = np.argsort(values)[::-1][:20]

    for rank, i in enumerate(indices, 1):

        row = rows[i]
        col = cols[i]

        x, y = rasterio.transform.xy(
            src.transform,
            row,
            col,
            offset="center"
        )

        print(
            f"{rank:2d}. "
            f"TRI={tri[row,col]:8.2f} | "
            f"X={x:10.2f} | "
            f"Y={y:10.2f} | "
            f"Boundary distance={distance[row,col]:8.2f} m"
        )

print("\nCHECK COMPLETE")