from pathlib import Path
import math

import numpy as np
import rasterio
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pyproj import Transformer


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "deep_resmlp_static.pt"

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

DEVICE = torch.device("cpu")


# ============================================================
# Deep ResMLP architecture
# MUST match train_deep.py
# ============================================================

class ResidualBlock(nn.Module):
    def __init__(self, features, dropout=0.20):
        super().__init__()

        self.block = nn.Sequential(
            nn.Linear(features, features),
            nn.BatchNorm1d(features),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(features, features),
            nn.BatchNorm1d(features),
        )

        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activation(x + self.block(x))


class DeepResMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
        )

        self.res1 = ResidualBlock(128, 0.20)
        self.res2 = ResidualBlock(128, 0.20)
        self.res3 = ResidualBlock(128, 0.20)

        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = self.input_layer(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        return self.head(x).squeeze(1)


# ============================================================
# Load Deep ResMLP ensemble
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Deep ResMLP model not found: {MODEL_PATH}"
    )

# This is a trusted local model file created by train_deep.py.
checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False,
)

saved_features = checkpoint.get("input_features")

if saved_features is not None:
    saved_features = list(saved_features)

    if saved_features != MODEL_FEATURES:
        raise RuntimeError(
            "Model feature order does not match backend.\n"
            f"Expected: {MODEL_FEATURES}\n"
            f"Model:    {saved_features}"
        )

SCALER_MEAN = np.asarray(
    checkpoint["scaler_mean"],
    dtype=np.float32,
)

SCALER_SCALE = np.asarray(
    checkpoint["scaler_scale"],
    dtype=np.float32,
)

THRESHOLD = float(
    checkpoint["threshold"]
)

STATE_DICTS = checkpoint["models"]

if len(STATE_DICTS) != 3:
    raise RuntimeError(
        f"Expected 3 ensemble models, found {len(STATE_DICTS)}."
    )

models = []

for state_dict in STATE_DICTS:
    net = DeepResMLP(
        input_dim=len(MODEL_FEATURES)
    ).to(DEVICE)

    net.load_state_dict(state_dict)
    net.eval()

    models.append(net)


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
    description=(
        "Static Deep Residual MLP Ensemble susceptibility "
        "inference for clicked map locations."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class LocationRequest(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)


# ============================================================
# Raster utilities
# ============================================================

def read_pixel(path: Path, x: float, y: float):
    """Read one raster pixel at projected coordinate x,y."""

    with rasterio.open(path) as src:

        if src.crs is None:
            raise RuntimeError(
                f"Raster has no CRS: {path}"
            )

        row, col = src.index(x, y)

        if not (
            0 <= row < src.height
            and 0 <= col < src.width
        ):
            raise ValueError(
                f"Point is outside raster extent: {path.name}"
            )

        value = src.read(
            1,
            window=((row, row + 1), (col, col + 1)),
        )[0, 0]

        nodata = src.nodata

        if nodata is not None and np.isclose(
            value,
            nodata
        ):
            raise ValueError(
                f"NoData at clicked location in {path.name}"
            )

        if not np.isfinite(value):
            raise ValueError(
                f"Invalid raster value in {path.name}"
            )

        return float(value)


# ============================================================
# Susceptibility classification
# ============================================================

def classify_susceptibility(probability: float) -> str:

    if probability < 0.25:
        return "LOW"

    if probability < 0.50:
        return "MODERATE"

    if probability < 0.75:
        return "HIGH"

    return "VERY HIGH"


# ============================================================
# Deep model inference
# ============================================================

def predict_probability(features):
    """
    Standardize the 13 features using the training scaler,
    run all three Deep ResMLP models, and average their
    probabilities.
    """

    values = np.asarray(
        [
            features[name]
            for name in MODEL_FEATURES
        ],
        dtype=np.float32,
    )

    # Same StandardScaler transformation used during training.
    scaled = (
        values - SCALER_MEAN
    ) / SCALER_SCALE

    tensor = torch.tensor(
        scaled,
        dtype=torch.float32,
        device=DEVICE,
    ).unsqueeze(0)

    probabilities = []

    with torch.no_grad():

        for net in models:

            logit = net(tensor)

            probability = torch.sigmoid(
                logit
            ).item()

            probabilities.append(
                probability
            )

    return float(
        np.mean(probabilities)
    )


# ============================================================
# Routes
# ============================================================

@app.get("/")
def root():

    return {
        "status": "online",
        "service": "Idukki Landslide Susceptibility API",
        "model": "Deep Residual MLP Ensemble",
        "model_type": "deep_learning",
        "ensemble_size": len(models),
        "classification_threshold": THRESHOLD,
    }


@app.get("/health")
def health():

    missing = [
        str(path)
        for path in RASTERS.values()
        if not path.exists()
    ]

    return {
        "status": "ok" if not missing else "error",
        "model_exists": MODEL_PATH.exists(),
        "model": "Deep Residual MLP Ensemble",
        "model_type": "deep_learning",
        "ensemble_size": len(models),
        "classification_threshold": THRESHOLD,
        "missing_rasters": missing,
        "features": MODEL_FEATURES,
    }


@app.post("/predict")
def predict(location: LocationRequest):

    lat = location.lat
    lon = location.lon

    # --------------------------------------------------------
    # Convert clicked WGS84 coordinates to UTM Zone 43N.
    # --------------------------------------------------------

    x, y = WGS84_TO_UTM43N.transform(
        lon,
        lat,
    )

    # --------------------------------------------------------
    # Extract raster values.
    # --------------------------------------------------------

    try:

        values = {
            "elevation": read_pixel(
                RASTERS["elevation"],
                x,
                y,
            ),

            "slope": read_pixel(
                RASTERS["slope"],
                x,
                y,
            ),

            "aspect": read_pixel(
                RASTERS["aspect"],
                x,
                y,
            ),

            "curvature": read_pixel(
                RASTERS["curvature"],
                x,
                y,
            ),

            "tri": read_pixel(
                RASTERS["tri"],
                x,
                y,
            ),

            "clay": read_pixel(
                RASTERS["clay"],
                x,
                y,
            ),

            "sand": read_pixel(
                RASTERS["sand"],
                x,
                y,
            ),

            "landcover": read_pixel(
                RASTERS["landcover"],
                x,
                y,
            ),
        }

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )


    # --------------------------------------------------------
    # Feature engineering
    # Same transformations used during training.
    # --------------------------------------------------------

    aspect_rad = math.radians(
        values["aspect"]
    )

    landcover_code = int(
        round(values["landcover"])
    )

    features = {

        "elevation":
            values["elevation"],

        "slope":
            values["slope"],

        "aspect_sin":
            math.sin(aspect_rad),

        "aspect_cos":
            math.cos(aspect_rad),

        "curvature":
            values["curvature"],

        "tri":
            values["tri"],

        "clay":
            values["clay"],

        "sand":
            values["sand"],

        "lc_10":
            int(landcover_code == 10),

        "lc_30":
            int(landcover_code == 30),

        "lc_40":
            int(landcover_code == 40),

        "lc_50":
            int(landcover_code == 50),

        "lc_80":
            int(landcover_code == 80),
    }


    # --------------------------------------------------------
    # Deep ResMLP ensemble prediction.
    # --------------------------------------------------------

    probability = predict_probability(
        features
    )

    # IMPORTANT:
    # 0.53 is the threshold selected using validation data
    # during Deep ResMLP training.
    prediction = int(
        probability >= THRESHOLD
    )

    landcover_seen_by_model = (
        landcover_code
        in {10, 30, 40, 50, 80}
    )


    # --------------------------------------------------------
    # Return response.
    #
    # The response structure is intentionally kept compatible
    # with the previous frontend/RF API.
    # --------------------------------------------------------

    return {

        "latitude":
            lat,

        "longitude":
            lon,

        "utm_x":
            round(x, 3),

        "utm_y":
            round(y, 3),

        "susceptibility":
            round(probability, 6),

        "risk_percent":
            round(probability * 100, 2),

        "risk_class":
            classify_susceptibility(
                probability
            ),

        "model_class":
            prediction,

        "elevation":
            round(
                values["elevation"],
                2,
            ),

        "slope":
            round(
                values["slope"],
                2,
            ),

        "aspect":
            round(
                values["aspect"] % 360,
                2,
            ),

        "curvature":
            round(
                values["curvature"],
                6,
            ),

        "tri":
            round(
                values["tri"],
                2,
            ),

        "clay":
            round(
                values["clay"],
                2,
            ),

        "sand":
            round(
                values["sand"],
                2,
            ),

        "landcover_code":
            landcover_code,

        "landcover":
            LANDCOVER_NAMES.get(
                landcover_code,
                f"Other / class {landcover_code}",
            ),

        "landcover_seen_by_model":
            landcover_seen_by_model,

        "model":
            "Deep Residual MLP Ensemble",

        "model_type":
            "deep_learning",

        "ensemble_size":
            len(models),

        "classification_threshold":
            THRESHOLD,

        "note": (
            "Static susceptibility only. This is not a "
            "rainfall-triggered landslide warning."
        ),
    }
