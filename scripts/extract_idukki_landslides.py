import geopandas as gpd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Paths
inventory_folder = PROJECT_ROOT / "datasets" / "raw" / "landslide_inventory"
boundary_path = PROJECT_ROOT / "datasets" / "raw" / "boundaries" / "idukki_boundary.gpkg"

output_path = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "landslides"
    / "idukki_landslides.gpkg"
)

output_path.parent.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Load landslide inventory
# ------------------------------------------------------------

shapefiles = list(inventory_folder.glob("*.shp"))

if not shapefiles:
    raise FileNotFoundError("No shapefile found.")

inventory_path = shapefiles[0]

print("Loading landslide inventory...")
landslides = gpd.read_file(inventory_path)

print("Total Kerala landslides:", len(landslides))

# ------------------------------------------------------------
# Extract Idukki using District attribute
# ------------------------------------------------------------

idukki = landslides[
    landslides["District"].str.strip().str.lower() == "idukki"
].copy()

print("Idukki landslides:", len(idukki))

# ------------------------------------------------------------
# Load Idukki boundary
# ------------------------------------------------------------

boundary = gpd.read_file(boundary_path)

print("Boundary CRS:", boundary.crs)

# Make sure both use WGS84
idukki = idukki.to_crs("EPSG:4326")
boundary = boundary.to_crs("EPSG:4326")

# ------------------------------------------------------------
# Spatial check
# ------------------------------------------------------------

inside = idukki.geometry.within(
    boundary.geometry.union_all()
)

print("Points inside Idukki boundary:", inside.sum())
print("Points outside boundary:", (~inside).sum())

# Keep only points physically inside the boundary
idukki = idukki[inside].copy()

# ------------------------------------------------------------
# Reproject to our DEM CRS
# ------------------------------------------------------------

idukki = idukki.to_crs("EPSG:32643")

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

idukki.to_file(
    output_path,
    driver="GPKG"
)

print("\n========== COMPLETE ==========")
print("Final Idukki landslides:", len(idukki))
print("CRS:", idukki.crs)
print("Output:", output_path)

print("\nColumns retained:")
print(list(idukki.columns))