from pathlib import Path
import math

import joblib
import numpy as np
import pandas as pd
import rasterio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pyproj import Transformer

# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "random_forest_static_v2.joblib"

RASTERS = {
    "elevation": PROJECT_ROOT / "datasets" / "processed" / "terrain_features" / "idukki_elevation.tif",
    "slope": PROJECT_ROOT / "datasets" / "processed" / "terrain_features" / "idukki_slope.tif",
    "aspect": PROJECT_ROOT / "datasets" / "processed" / "terrain_features" / "idukki_aspect.tif",
    "curvature": PROJECT_ROOT / "datasets" / "processed" / "terrain_features" / "idukki_curvature.tif",
    "tri": PROJECT_ROOT / "datasets" / "processed" / "terrain_features" / "idukki_tri.tif",
    "clay": PROJECT_ROOT / "datasets" / "processed" / "soil" / "idukki_clay_0_5cm.tif",
    "sand": PROJECT_ROOT / "datasets" / "processed" / "soil" / "idukki_sand_0_5cm.tif",
    "landcover": PROJECT_ROOT / "datasets" / "processed" / "landcover" / "idukki_worldcover2021.tif",
}

MODEL_FEATURES = [
    "elevation",
    "slope",
    "aspect_sin",
    "aspect_cos",
    "curvature",
    "tri",
    "clay",
    "sand",
    "lc_10",
    "lc_30",
    "lc_40",
    "lc_50",
    "lc_80",
]

LANDCOVER_NAMES = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / sparse vegetation",
    70: "Snow / ice",
    80: "Permanent water",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss / lichen",
}

# ============================================================
# Load model
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

# Make sure this backend uses the exact feature order used during training.
if hasattr(model, "feature_names_in_"):
    trained_features = list(model.feature_names_in_)
    if trained_features != MODEL_FEATURES:
        raise RuntimeError(
            "Model feature order does not match the backend.\n"
            f"Expected: {MODEL_FEATURES}\n"
            f"Model:    {trained_features}"
        )

# ============================================================
# Coordinate transformation
# ============================================================

WGS84_TO_UTM43N = Transformer.from_crs(
    "EPSG:4326",
    "EPSG:32643",
    always_xy=True,
)

# ============================================================
# FastAPI
# ============================================================

app = FastAPI(
    title="Idukki Landslide Susceptibility API",
    description="Static Random Forest susceptibility inference for clicked map locations.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local prototype; restrict this later for deployment
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class LocationRequest(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)


def read_pixel(path: Path, x: float, y: float):
    """Read one raster pixel at projected coordinate x,y."""
    with rasterio.open(path) as src:
        if src.crs is None:
            raise RuntimeError(f"Raster has no CRS: {path}")

        row, col = src.index(x, y)

        if not (0 <= row < src.height and 0 <= col < src.width):
            raise ValueError(f"Point is outside raster extent: {path.name}")

        value = src.read(1, window=((row, row + 1), (col, col + 1)))[0, 0]

        nodata = src.nodata
        if nodata is not None and np.isclose(value, nodata):
            raise ValueError(f"NoData at clicked location in {path.name}")

        if not np.isfinite(value):
            raise ValueError(f"Invalid raster value in {path.name}")

        return float(value)


def classify_susceptibility(probability: float) -> str:
    """Same display bins used by the web susceptibility map."""
    if probability < 0.25:
        return "LOW"
    if probability < 0.50:
        return "MODERATE"
    if probability < 0.75:
        return "HIGH"
    return "VERY HIGH"


@app.get("/")
def root():
    return {
        "status": "online",
        "service": "Idukki Landslide Susceptibility API",
        "model": "Random Forest static v2",
    }


@app.get("/health")
def health():
    missing = [str(path) for path in RASTERS.values() if not path.exists()]

    return {
        "status": "ok" if not missing else "error",
        "model_exists": MODEL_PATH.exists(),
        "missing_rasters": missing,
        "features": MODEL_FEATURES,
    }


@app.post("/predict")
def predict(location: LocationRequest):
    lat = location.lat
    lon = location.lon

    # Convert clicked WGS84 coordinates to the raster CRS.
    x, y = WGS84_TO_UTM43N.transform(lon, lat)

    try:
        values = {
            "elevation": read_pixel(RASTERS["elevation"], x, y),
            "slope": read_pixel(RASTERS["slope"], x, y),
            "aspect": read_pixel(RASTERS["aspect"], x, y),
            "curvature": read_pixel(RASTERS["curvature"], x, y),
            "tri": read_pixel(RASTERS["tri"], x, y),
            "clay": read_pixel(RASTERS["clay"], x, y),
            "sand": read_pixel(RASTERS["sand"], x, y),
            "landcover": read_pixel(RASTERS["landcover"], x, y),
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    aspect_rad = math.radians(values["aspect"])

    landcover_code = int(round(values["landcover"]))

    features = {
        "elevation": values["elevation"],
        "slope": values["slope"],
        "aspect_sin": math.sin(aspect_rad),
        "aspect_cos": math.cos(aspect_rad),
        "curvature": values["curvature"],
        "tri": values["tri"],
        "clay": values["clay"],
        "sand": values["sand"],
        "lc_10": int(landcover_code == 10),
        "lc_30": int(landcover_code == 30),
        "lc_40": int(landcover_code == 40),
        "lc_50": int(landcover_code == 50),
        "lc_80": int(landcover_code == 80),
    }

    # Use a DataFrame with the exact feature names/order used in training.
    X = pd.DataFrame([features], columns=MODEL_FEATURES)

    probability = float(model.predict_proba(X)[0, 1])
    prediction = int(model.predict(X)[0])

    # Some ESA classes were not represented in the v2 training samples.
    landcover_seen_by_model = landcover_code in {10, 30, 40, 50, 80}

    return {
        "latitude": lat,
        "longitude": lon,
        "utm_x": round(x, 3),
        "utm_y": round(y, 3),

        "susceptibility": round(probability, 6),
        "risk_percent": round(probability * 100, 2),
        "risk_class": classify_susceptibility(probability),
        "model_class": prediction,

        "elevation": round(values["elevation"], 2),
        "slope": round(values["slope"], 2),
        "aspect": round(values["aspect"] % 360, 2),
        "curvature": round(values["curvature"], 6),
        "tri": round(values["tri"], 2),
        "clay": round(values["clay"], 2),
        "sand": round(values["sand"], 2),

        "landcover_code": landcover_code,
        "landcover": LANDCOVER_NAMES.get(
            landcover_code,
            f"Other / class {landcover_code}",
        ),
        "landcover_seen_by_model": landcover_seen_by_model,

        "model": "Random Forest static v2",
        "note": (
            "Static susceptibility only. This is not a rainfall-triggered "
            "landslide warning."
        ),
    }
