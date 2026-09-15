import geopandas as gpd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

folder = PROJECT_ROOT / "datasets" / "raw" / "landslide_inventory"

# Automatically find the shapefile
shapefiles = list(folder.glob("*.shp"))

if not shapefiles:
    raise FileNotFoundError("No .shp file found in landslide_inventory folder.")

shp_path = shapefiles[0]

print("Loading:", shp_path.name)

gdf = gpd.read_file(shp_path)

print("\n========== LANDSLIDE INVENTORY ==========")
print("Number of records:", len(gdf))
print("CRS:", gdf.crs)
print("Geometry type:", gdf.geometry.geom_type.value_counts().to_dict())

print("\nColumns:")
for column in gdf.columns:
    print(" -", column)

print("\nFirst 5 records:")
print(gdf.head())