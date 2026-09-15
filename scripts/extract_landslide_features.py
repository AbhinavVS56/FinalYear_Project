import geopandas as gpd
import rasterio
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LANDSLIDE_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "landslides"
    / "idukki_landslides.gpkg"
)

FEATURE_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "terrain_features"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "training"
    / "idukki_positive_samples.csv"
)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Load landslides
# ------------------------------------------------------------

print("Loading Idukki landslides...")

landslides = gpd.read_file(LANDSLIDE_PATH)

print("Landslide points:", len(landslides))
print("CRS:", landslides.crs)


# ------------------------------------------------------------
# Feature files
# ------------------------------------------------------------

features = {
    "elevation": FEATURE_DIR / "idukki_elevation.tif",
    "slope": FEATURE_DIR / "idukki_slope.tif",
    "aspect": FEATURE_DIR / "idukki_aspect.tif",
    "curvature": FEATURE_DIR / "idukki_curvature.tif",
    "tri": FEATURE_DIR / "idukki_tri.tif"
}


# ------------------------------------------------------------
# Make sure points use DEM CRS
# ------------------------------------------------------------

with rasterio.open(features["elevation"]) as src:

    landslides = landslides.to_crs(src.crs)

    coordinates = [
        (point.x, point.y)
        for point in landslides.geometry
    ]


# ------------------------------------------------------------
# Extract raster values
# ------------------------------------------------------------

print("\nExtracting terrain values...")

data = {}

for name, path in features.items():

    print("Reading:", name)

    with rasterio.open(path) as src:

        values = list(src.sample(coordinates))

        values = np.array(values).flatten()

        data[name] = values


# ------------------------------------------------------------
# Create dataframe
# ------------------------------------------------------------

df = pd.DataFrame(data)

# Positive class
df["label"] = 1


# ------------------------------------------------------------
# Check missing/invalid values
# ------------------------------------------------------------

print("\n========== DATA CHECK ==========")

print("Rows:", len(df))

print("\nMissing values:")
print(df.isna().sum())

print("\nFeature statistics:")
print(df.describe())


# ------------------------------------------------------------
# Remove invalid rows
# ------------------------------------------------------------

df = df.replace([np.inf, -np.inf], np.nan)

df = df.dropna()

print("\nRows after cleaning:", len(df))


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

df.to_csv(OUTPUT_PATH, index=False)

print("\n========== COMPLETE ==========")
print("Positive samples:", len(df))
print("Saved to:")
print(OUTPUT_PATH)