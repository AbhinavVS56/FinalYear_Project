import os
import joblib
import numpy as np
import rasterio
from rasterio.warp import transform
import math

# --------------------------------------------------
# FILE PATHS
# --------------------------------------------------

MODEL_PATH = "models/random_forest_static_v2.joblib"

RASters = {
    "elevation": "datasets/processed/terrain_features/idukki_elevation.tif",
    "slope": "datasets/processed/terrain_features/idukki_slope.tif",
    "aspect": "datasets/processed/terrain_features/idukki_aspect.tif",
    "curvature": "datasets/processed/terrain_features/idukki_curvature.tif",
    "tri": "datasets/processed/terrain_features/idukki_tri.tif",
    "clay": "datasets/processed/soil/idukki_clay_0_5cm.tif",
    "sand": "datasets/processed/soil/idukki_sand_0_5cm.tif",
    "landcover": "datasets/processed/landcover/idukki_worldcover2021.tif"
}


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

print("\nLoading Random Forest model...")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully.")


# --------------------------------------------------
# GET USER LOCATION
# --------------------------------------------------

latitude = float(input("\nEnter latitude: "))
longitude = float(input("Enter longitude: "))


# --------------------------------------------------
# OPEN DEM TO GET TARGET GRID / CRS
# --------------------------------------------------

with rasterio.open(RASters["elevation"]) as src:

    raster_crs = src.crs
    transform_affine = src.transform

    # Convert WGS84 latitude/longitude to raster CRS
    x, y = transform(
        "EPSG:4326",
        raster_crs,
        [longitude],
        [latitude]
    )

    x = x[0]
    y = y[0]

    row, col = rasterio.transform.rowcol(
        transform_affine,
        x,
        y
    )

    # Check whether point is inside raster
    if row < 0 or row >= src.height or col < 0 or col >= src.width:
        print("\nERROR: Location is outside the Idukki study area.")
        exit()

    elevation = src.read(1)[row, col]


# --------------------------------------------------
# FUNCTION TO READ VALUE FROM SAME PIXEL
# --------------------------------------------------

def get_value(path, row, col):

    with rasterio.open(path) as src:

        value = src.read(1)[row, col]

        return float(value)


# --------------------------------------------------
# EXTRACT TERRAIN FEATURES
# --------------------------------------------------

slope = get_value(RASters["slope"], row, col)
aspect = get_value(RASters["aspect"], row, col)
curvature = get_value(RASters["curvature"], row, col)
tri = get_value(RASters["tri"], row, col)

# Aspect is circular → convert to sine/cosine
aspect_rad = math.radians(aspect)

aspect_sin = math.sin(aspect_rad)
aspect_cos = math.cos(aspect_rad)


# --------------------------------------------------
# EXTRACT SOIL FEATURES
# --------------------------------------------------

clay = get_value(RASters["clay"], row, col)
sand = get_value(RASters["sand"], row, col)


# --------------------------------------------------
# EXTRACT LANDCOVER
# --------------------------------------------------

landcover = int(get_value(RASters["landcover"], row, col))


# --------------------------------------------------
# LANDCOVER ONE-HOT ENCODING
# --------------------------------------------------

lc_10 = 1 if landcover == 10 else 0
lc_30 = 1 if landcover == 30 else 0
lc_40 = 1 if landcover == 40 else 0
lc_50 = 1 if landcover == 50 else 0
lc_80 = 1 if landcover == 80 else 0


# --------------------------------------------------
# CREATE MODEL INPUT
# EXACT SAME ORDER AS TRAINING
# --------------------------------------------------

features = np.array([[
    elevation,
    slope,
    aspect_sin,
    aspect_cos,
    curvature,
    tri,
    clay,
    sand,
    lc_10,
    lc_30,
    lc_40,
    lc_50,
    lc_80
]])


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

prediction = model.predict(features)[0]

probabilities = model.predict_proba(features)[0]

# Find probability corresponding to class 1
class_1_index = list(model.classes_).index(1)

susceptibility = probabilities[class_1_index] * 100


# --------------------------------------------------
# CLASSIFICATION
# --------------------------------------------------

if susceptibility < 30:
    risk = "LOW"
elif susceptibility < 60:
    risk = "MODERATE"
elif susceptibility < 80:
    risk = "HIGH"
else:
    risk = "VERY HIGH"


# --------------------------------------------------
# LANDCOVER NAME
# --------------------------------------------------

landcover_names = {
    10: "Tree Cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / Sparse Vegetation",
    70: "Snow / Ice",
    80: "Permanent Water Bodies",
    90: "Herbaceous Wetland",
    95: "Mangroves",
    100: "Moss / Lichen"
}

landcover_name = landcover_names.get(
    landcover,
    "Unknown"
)


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

print("\n" + "=" * 50)
print("        LANDSLIDE SUSCEPTIBILITY")
print("=" * 50)

print(f"\nLatitude       : {latitude:.6f}")
print(f"Longitude      : {longitude:.6f}")

print("\nEnvironmental Features")
print("-" * 30)

print(f"Elevation      : {elevation:.2f} m")
print(f"Slope          : {slope:.2f}°")
print(f"Aspect         : {aspect:.2f}°")
print(f"Curvature      : {curvature:.6f}")
print(f"TRI            : {tri:.2f}")
print(f"Clay           : {clay:.2f}%")
print(f"Sand           : {sand:.2f}%")
print(f"Landcover      : {landcover_name} ({landcover})")

print("\nPrediction")
print("-" * 30)

print(f"Susceptibility : {susceptibility:.2f}%")
print(f"Risk Level     : {risk}")
print(f"Class          : {prediction}")

print("\n" + "=" * 50)