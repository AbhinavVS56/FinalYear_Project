import random
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


# ============================================================
# CONFIG
# ============================================================

SEED = 42

PATCH_SIZE = 13
HALF = PATCH_SIZE // 2

BLOCK_SIZE = 2000

BATCH_SIZE = 128
EPOCHS = 128

# Paper-aligned MSCNN is sensitive to learning rate.
# This is an adaptation for our Idukki dataset, not a claim of
# reproducing the paper's exact training run.
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 5e-4
DROPOUT = 0.20

PATIENCE = 20

DEVICE = torch.device("cpu")


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "training"
    / "idukki_training_dataset_v2.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"
RESULT_DIR = PROJECT_ROOT / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# MSCNN V6 - PAPER-ALIGNED STATIC MODEL
#
# The base paper's static MSCNN uses three 3x3 convolutional
# layers with depths 64, 128 and 256 and fuses representations
# from different convolutional depths.
#
# This implementation adapts that design to our Idukki dataset:
#   input: 13 x 13 patch, 15 channels
#   Conv 3x3: 13 -> 64
#   Conv 3x3: 64 -> 128
#   Conv 3x3: 128 -> 256
#   multi-depth feature fusion: 64 + 128 + 256 channels
#   compact fusion + classifier
#
# Important: the architecture is paper-aligned, but the Idukki
# data, 13 input factors and spatial validation protocol differ
# from the paper's datasets/protocol.
# ============================================================

class MSCNN(nn.Module):
    """
    MSCNN V6:
    Paper-aligned 3x3 MSCNN (64/128/256) plus a compact center-feature branch.
    Global average/max pooling reduces the overfitting seen in V4.
    """
    def __init__(self, in_channels=15):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Concatenate features from all three depths: 64 + 128 + 256 = 448.
        self.fuse = nn.Sequential(
            nn.Conv2d(448, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Center pixel contains the same 15 static factors used by RF.
        self.center_branch = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.30),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True)
        )

        # 128 avg + 128 max + 32 center = 288 features.
        self.classifier = nn.Sequential(
            nn.Linear(288, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.30),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.20),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)

        fused = self.fuse(torch.cat([x1, x2, x3], dim=1))

        avg_features = self.avg_pool(fused).flatten(1)
        max_features = self.max_pool(fused).flatten(1)

        center = x[:, :, x.shape[2] // 2, x.shape[3] // 2]
        center_features = self.center_branch(center)

        combined = torch.cat(
            [avg_features, max_features, center_features], dim=1
        )
        return self.classifier(combined).squeeze(1)


# ============================================================

print("\n" + "=" * 70)
print("MSCNN V6 - HYBRID STATIC LANDSLIDE SUSCEPTIBILITY")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(DATASET)

print("Dataset shape:", df.shape)
print("Columns:")
print(list(df.columns))


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "elevation",
    "slope",
    "aspect",
    "curvature",
    "tri",
    "x",
    "y",
    "label",
    "clay",
    "sand",
    "landcover",
]

missing = [c for c in required_columns if c not in df.columns]

if missing:
    raise ValueError(f"Missing dataset columns: {missing}")


# ============================================================
# ============================================================
# RASTER PATHS
# ============================================================

TERRAIN_DIR = PROJECT_ROOT / "datasets" / "processed" / "terrain_features"
SOIL_DIR = PROJECT_ROOT / "datasets" / "processed" / "soil"
LANDCOVER_DIR = PROJECT_ROOT / "datasets" / "processed" / "landcover"

DEM_PATH = TERRAIN_DIR / "idukki_elevation.tif"
TWI_PATH = TERRAIN_DIR / "idukki_twi.tif"
SPI_PATH = TERRAIN_DIR / "idukki_spi.tif"

raster_paths = {
    "elevation": DEM_PATH,
    "slope": TERRAIN_DIR / "idukki_slope.tif",
    "aspect": TERRAIN_DIR / "idukki_aspect.tif",
    "curvature": TERRAIN_DIR / "idukki_curvature.tif",
    "tri": TERRAIN_DIR / "idukki_tri.tif",
    "twi": TWI_PATH,
    "spi": SPI_PATH,
    "clay": SOIL_DIR / "idukki_clay_0_5cm.tif",
    "sand": SOIL_DIR / "idukki_sand_0_5cm.tif",
    "landcover": LANDCOVER_DIR / "idukki_worldcover2021.tif",
}


def create_twi_spi():
    """
    Create DEM-derived terrain wetness/runoff factors.

    TWI/SPI require contributing area. A full hydrological flow-routing
    implementation would be considerably heavier on CPU, so this project
    uses a stable local contributing-area proxy derived from DEM relief.

    TWI = ln(a / tan(slope))
    SPI = a * tan(slope)

    These are modelling factors derived from the existing Copernicus DEM;
    they should be described as DEM-derived approximations in the report.
    """
    if TWI_PATH.exists() and SPI_PATH.exists():
        print("\nTWI/SPI rasters already exist.")
        return

    print("\nCreating DEM-derived TWI/SPI rasters...")

    from scipy.ndimage import uniform_filter

    with rasterio.open(DEM_PATH) as src:
        dem = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        nodata = src.nodata

        valid = np.isfinite(dem)
        if nodata is not None:
            valid &= dem != nodata

        if not np.any(valid):
            raise ValueError("DEM contains no valid pixels.")

        fill_value = float(np.nanmedian(dem[valid]))
        work = dem.copy()
        work[~valid] = fill_value

        # Broad local relief as a stable contributing-area proxy.
        mean_dem = uniform_filter(work, size=21, mode="nearest")
        relief = np.maximum(mean_dem - work, 0.0)

        # Keep the proxy strictly positive.
        cell_size = float(np.mean(src.res))
        contributing_area = relief + (2.0 * cell_size)

        gy, gx = np.gradient(work, src.res[1], src.res[0])
        slope_rad = np.arctan(np.sqrt(gx * gx + gy * gy))
        slope_rad = np.clip(
            slope_rad,
            np.deg2rad(0.1),
            np.deg2rad(89.0)
        )

        tan_slope = np.tan(slope_rad)

        twi = np.log(
            np.maximum(contributing_area / tan_slope, 1e-6)
        ).astype(np.float32)

        spi = (
            contributing_area * tan_slope
        ).astype(np.float32)

        twi[~valid] = np.nan
        spi[~valid] = np.nan

        profile.update(
            dtype="float32",
            count=1,
            nodata=np.nan,
            compress="lzw"
        )

        TWI_PATH.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(TWI_PATH, "w", **profile) as dst:
            dst.write(twi, 1)

        with rasterio.open(SPI_PATH, "w", **profile) as dst:
            dst.write(spi, 1)

    print("TWI saved:", TWI_PATH)
    print("SPI saved:", SPI_PATH)


create_twi_spi()

for name, path in raster_paths.items():
    if not path.exists():
        raise FileNotFoundError(f"Missing raster: {path}")

print("\nAll required rasters found.")

# ============================================================
# SPATIAL BLOCKS
# ============================================================

print("\nCreating 2 km spatial blocks...")

df["block_x"] = np.floor(df["x"] / BLOCK_SIZE).astype(int)
df["block_y"] = np.floor(df["y"] / BLOCK_SIZE).astype(int)

groups = df["block_x"].astype(str) + "_" + df["block_y"].astype(str)

print("Spatial blocks:", groups.nunique())


# ============================================================
# OUTER SPATIAL TRAIN / TEST SPLIT
# ============================================================

print("\nCreating spatial train/test split...")

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)

train_idx, test_idx = next(sgkf.split(df, df["label"], groups=groups))

train_full = df.iloc[train_idx].reset_index(drop=True)
test_df = df.iloc[test_idx].reset_index(drop=True)

print("\nTraining pool:", len(train_full))
print("Test set:", len(test_df))


# ============================================================
# INNER TRAIN / VALIDATION SPLIT
# ============================================================

print("\nCreating validation split...")

train_groups = train_full["block_x"].astype(str) + "_" + train_full["block_y"].astype(str)

inner_sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED + 1)

inner_train_idx, val_idx = next(
    inner_sgkf.split(train_full, train_full["label"], groups=train_groups)
)

train_df = train_full.iloc[inner_train_idx].reset_index(drop=True)
val_df = train_full.iloc[val_idx].reset_index(drop=True)

print("Training samples:", len(train_df))
print("Validation samples:", len(val_df))
print("Testing samples:", len(test_df))

print("\nTraining distribution:")
print(train_df["label"].value_counts())

print("\nValidation distribution:")
print(val_df["label"].value_counts())

print("\nTesting distribution:")
print(test_df["label"].value_counts())


# ============================================================
# OPEN RASTERS
# ============================================================

print("\nOpening rasters...")

rasters = {name: rasterio.open(path) for name, path in raster_paths.items()}
reference = rasters["elevation"]

print("\nRaster:")
print("Width :", reference.width)
print("Height:", reference.height)
print("CRS   :", reference.crs)


# ============================================================
# LANDCOVER
# ============================================================

LANDCOVER_CLASSES = [10, 30, 40, 50, 80]


# ============================================================
# PATCH EXTRACTION
# ============================================================

def extract_patch(row):

    x = float(row["x"])
    y = float(row["y"])

    col, row_pixel = reference.index(x, y)

    window = Window(col - HALF, row_pixel - HALF, PATCH_SIZE, PATCH_SIZE)

    arrays = {}

    for name, raster in rasters.items():
        arr = raster.read(
            1, window=window, boundless=True, fill_value=np.nan
        ).astype(np.float32)
        arrays[name] = arr

    elevation = arrays["elevation"]
    slope = arrays["slope"]
    aspect = arrays["aspect"]
    curvature = arrays["curvature"]
    tri = arrays["tri"]
    twi = arrays["twi"]
    spi = arrays["spi"]
    clay = arrays["clay"]
    sand = arrays["sand"]
    landcover = arrays["landcover"]

    # --------------------------------------------------------
    # Aspect circular encoding
    # --------------------------------------------------------
    aspect_rad = np.radians(aspect)
    aspect_sin = np.sin(aspect_rad).astype(np.float32)
    aspect_cos = np.cos(aspect_rad).astype(np.float32)

    # --------------------------------------------------------
    # Landcover one-hot
    # --------------------------------------------------------
    lc_channels = [
        (landcover == cls).astype(np.float32) for cls in LANDCOVER_CLASSES
    ]

    # --------------------------------------------------------
    # Stack -- order must match RF v2 feature order exactly:
    # elevation, slope, aspect_sin, aspect_cos, curvature, tri,
    # twi, spi, clay, sand, lc_10, lc_30, lc_40, lc_50, lc_80
    # --------------------------------------------------------
    patch = np.stack(
        [elevation, slope, aspect_sin, aspect_cos, curvature, tri,
         twi, spi, clay, sand, *lc_channels],
        axis=0,
    )

    # --------------------------------------------------------
    # Replace invalid values with the center pixel value
    # instead of discarding the sample.
    # --------------------------------------------------------
    for channel in range(10):
        center_value = patch[channel, HALF, HALF]

        if not np.isfinite(center_value):
            center_value = 0.0

        invalid = ~np.isfinite(patch[channel])
        patch[channel][invalid] = center_value

    patch[8:] = np.nan_to_num(patch[8:], nan=0.0, posinf=0.0, neginf=0.0)
    patch = np.nan_to_num(patch, nan=0.0, posinf=0.0, neginf=0.0)

    return patch


# ============================================================
# BUILD PATCH DATASET
# ============================================================

def build_patches(dataframe, name):

    print(f"\nExtracting {name} patches...")

    patches = []
    labels = []
    total = len(dataframe)

    for i, (_, row) in enumerate(dataframe.iterrows(), start=1):
        patch = extract_patch(row)
        patches.append(patch)
        labels.append(int(row["label"]))

        if i % 500 == 0:
            print(f"Processed {i}/{total}")

    patches = np.stack(patches).astype(np.float32)
    labels = np.array(labels, dtype=np.float32)

    print(f"{name} patches:", patches.shape)
    print(f"{name} labels:", np.bincount(labels.astype(int)))

    return patches, labels


X_train, y_train = build_patches(train_df, "training")
X_val, y_val = build_patches(val_df, "validation")
X_test, y_test = build_patches(test_df, "testing")


# ============================================================
# NORMALIZATION
# ============================================================

print("\nNormalizing continuous channels...")

CONTINUOUS_CHANNELS = 10

means = X_train[:, :CONTINUOUS_CHANNELS].mean(axis=(0, 2, 3))
stds = X_train[:, :CONTINUOUS_CHANNELS].std(axis=(0, 2, 3))
stds[stds < 1e-6] = 1.0


def normalize(X):
    X[:, :CONTINUOUS_CHANNELS] = (
        X[:, :CONTINUOUS_CHANNELS] - means.reshape(1, -1, 1, 1)
    ) / stds.reshape(1, -1, 1, 1)
    return X


X_train = normalize(X_train)
X_val = normalize(X_val)
X_test = normalize(X_test)


# ============================================================
# TORCH DATASETS
# ============================================================

train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


# ============================================================
# MODEL
# ============================================================

print("\nCreating MSCNN V6 (paper-aligned)...")

model = MSCNN(in_channels=15).to(DEVICE)

print(model)
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")


# ============================================================
# LOSS
# ============================================================

# The Idukki training split is already close to balanced, so do not
# artificially up-weight the positive class.
criterion = nn.BCEWithLogitsLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6
)


# ============================================================
# EVALUATION
# ============================================================

def evaluate(loader):

    model.eval()

    probabilities = []
    targets = []
    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(inputs)
            loss = criterion(logits, labels)
            probs = torch.sigmoid(logits)

            probabilities.extend(probs.cpu().numpy())
            targets.extend(labels.cpu().numpy())

            total_loss += loss.item() * len(labels)
            total_count += len(labels)

    probabilities = np.array(probabilities)
    targets = np.array(targets)
    predictions = (probabilities >= 0.5).astype(int)

    accuracy = accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions, zero_division=0)
    recall = recall_score(targets, predictions, zero_division=0)
    f1 = f1_score(targets, predictions, zero_division=0)
    auc = roc_auc_score(targets, probabilities)
    loss = total_loss / max(total_count, 1)

    return loss, accuracy, precision, recall, f1, auc, probabilities, predictions, targets


# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 70)
print("STARTING MSCNN V6 TRAINING")
print("=" * 70)

best_auc = -1.0
best_epoch = 0
patience_counter = 0

model_path = MODEL_DIR / "mscnn_static_v6.pt"

for epoch in range(1, EPOCHS + 1):

    model.train()

    running_loss = 0.0
    samples_seen = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        running_loss += loss.item() * len(labels)
        samples_seen += len(labels)

    train_loss = running_loss / max(samples_seen, 1)

    (val_loss, val_accuracy, val_precision, val_recall, val_f1, val_auc, _, _, _) = evaluate(val_loader)

    scheduler.step(val_auc)

    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Epoch {epoch:03d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy:.4f} | "
        f"Val F1: {val_f1:.4f} | "
        f"Val AUC: {val_auc:.4f} | "
        f"LR: {current_lr:.6f}"
    )

    if val_auc > best_auc:
        best_auc = val_auc
        best_epoch = epoch
        patience_counter = 0

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "means": means,
                "stds": stds,
                "patch_size": PATCH_SIZE,
                "channels": 13,
                "best_val_auc": best_auc,
            },
            model_path,
        )

        print(f"  -> Best model saved (Val AUC={val_auc:.4f})")
    else:
        patience_counter += 1

    if patience_counter >= PATIENCE:
        print("\nEarly stopping.")
        break


# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\nLoading best MSCNN...")

checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

(test_loss, test_accuracy, test_precision, test_recall, test_f1, test_auc,
 probabilities, predictions, targets) = evaluate(test_loader)

cm = confusion_matrix(targets, predictions)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FINAL MSCNN V6 RESULTS")
print("=" * 70)

print(f"Best Epoch: {best_epoch}")
print(f"Validation AUC: {best_auc:.4f}")
print(f"Accuracy : {test_accuracy:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall   : {test_recall:.4f}")
print(f"F1 Score : {test_f1:.4f}")
print(f"ROC-AUC  : {test_auc:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        targets, predictions, target_names=["Non-Landslide", "Landslide"], zero_division=0
    )
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics_path = RESULT_DIR / "mscnn_static_v6_metrics.txt"

with open(metrics_path, "w") as f:
    f.write("MSCNN V6 PAPER-ALIGNED STATIC LANDSLIDE SUSCEPTIBILITY\n")
    f.write("=" * 60 + "\n\n")

    f.write(f"Dataset size: {len(df)}\n")
    f.write(f"Training samples: {len(train_df)}\n")
    f.write(f"Validation samples: {len(val_df)}\n")
    f.write(f"Testing samples: {len(test_df)}\n")
    f.write(f"Patch size: {PATCH_SIZE}x{PATCH_SIZE}\n")
    f.write("Input channels: 13\n")
    f.write(f"Spatial block size: {BLOCK_SIZE} m\n\n")

    f.write(f"Best epoch: {best_epoch}\n")
    f.write(f"Validation AUC: {best_auc:.4f}\n\n")

    f.write(f"Accuracy : {test_accuracy:.4f}\n")
    f.write(f"Precision: {test_precision:.4f}\n")
    f.write(f"Recall   : {test_recall:.4f}\n")
    f.write(f"F1 Score : {test_f1:.4f}\n")
    f.write(f"ROC-AUC  : {test_auc:.4f}\n\n")

    f.write("Confusion Matrix:\n")
    f.write(str(cm))

    f.write("\n\nClassification Report:\n")
    f.write(
        classification_report(
            targets, predictions, target_names=["Non-Landslide", "Landslide"], zero_division=0
        )
    )


# ============================================================
# CLOSE RASTERS
# ============================================================

for raster in rasters.values():
    raster.close()


# ============================================================
# COMPLETE
# ============================================================

print("\nModel saved to:")
print(model_path)

print("\nMetrics saved to:")
print(metrics_path)

print("\n" + "=" * 70)
print("MSCNN V6 TRAINING COMPLETE")
print("=" * 70)