import geopandas as gpd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

input_path = PROJECT_ROOT / "datasets" / "raw" / "boundaries" / "gadm41_IND_2.shp"
output_path = PROJECT_ROOT / "datasets" / "raw" / "boundaries" / "idukki_boundary.gpkg"

print("Loading India district boundaries...")

gdf = gpd.read_file(input_path)

print("Total districts:", len(gdf))
print("Columns:", list(gdf.columns))

# Find Idukki
idukki = gdf[
    gdf["NAME_2"].str.lower().str.strip() == "idukki"
].copy()

if idukki.empty:
    print("\nIdukki was not found.")
    print("Available Kerala districts:")
    kerala = gdf[gdf["NAME_1"].str.lower().str.strip() == "kerala"]
    print(kerala["NAME_2"].tolist())
    raise SystemExit

# Save boundary
idukki.to_file(output_path, driver="GPKG")

print("\n========== IDUKKI BOUNDARY ==========")
print("District:", idukki["NAME_2"].iloc[0])
print("State:", idukki["NAME_1"].iloc[0])
print("CRS:", idukki.crs)
print("Bounds:", idukki.total_bounds)
print("Area:", idukki.to_crs("EPSG:32643").area.iloc[0] / 1_000_000, "km²")
print("\nSaved to:")
print(output_path)