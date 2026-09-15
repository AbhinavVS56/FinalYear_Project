import rasterio
import geopandas as gpd
from rasterio.mask import mask
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

dem_path = PROJECT_ROOT / "datasets" / "raw" / "dem" / "cop30" / "idukki_cop30_dem.tif"

boundary_path = PROJECT_ROOT / "datasets" / "raw" / "boundaries" / "idukki_boundary.gpkg"

output_path = PROJECT_ROOT / "datasets" / "processed" / "dem" / "idukki_dem_clipped.tif"

output_path.parent.mkdir(parents=True, exist_ok=True)

print("Loading Idukki boundary...")
boundary = gpd.read_file(boundary_path)

print("Boundary CRS:", boundary.crs)

with rasterio.open(dem_path) as dem:

    print("\nLoading DEM...")
    print("DEM CRS:", dem.crs)

    # Make sure boundary uses the same CRS as DEM
    boundary = boundary.to_crs(dem.crs)

    print("Clipping DEM...")

    clipped, transform = mask(
        dem,
        boundary.geometry,
        crop=True
    )

    profile = dem.profile.copy()

    profile.update({
        "height": clipped.shape[1],
        "width": clipped.shape[2],
        "transform": transform
    })

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(clipped)

print("\n========== CLIPPING COMPLETE ==========")
print("Output:", output_path)
print("Shape:", clipped.shape)
print("Saved successfully.")