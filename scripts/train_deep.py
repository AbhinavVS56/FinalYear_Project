import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


# ============================================================================
# CONFIG
# ============================================================================

SEED = 42
BLOCK_SIZE = 2000
BATCH_SIZE = 128
EPOCHS = 200
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 25
N_MODELS = 3

DEVICE = torch.device("cpu")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
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

MODEL_PATH = MODEL_DIR / "deep_resmlp_static.pt"
METRICS_PATH = RESULT_DIR / "deep_resmlp_static_metrics.txt"


# ============================================================================
# REPRODUCIBILITY
# ============================================================================

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


seed_everything(SEED)


print("=" * 70)
print("DEEP RESMLP - STATIC LANDSLIDE SUSCEPTIBILITY")
print("=" * 70)
print()
print("Device:", DEVICE)
print("Dataset:", DATASET_PATH)


# ============================================================================
# LOAD DATA
# ============================================================================

print("\nLoading dataset...")

df = pd.read_csv(DATASET_PATH)

required = [
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

missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

print("Dataset shape:", df.shape)
print("Columns:", list(df.columns))


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

# Aspect is circular. Convert it to sin/cos rather than treating 0° and 360°
# as distant values.
df["aspect_sin"] = np.sin(np.deg2rad(df["aspect"]))
df["aspect_cos"] = np.cos(np.deg2rad(df["aspect"]))

# Landcover is categorical. Use the same WorldCover classes used by the RF
# baseline, but feed them to the neural network as one-hot features.
landcover_classes = [10, 30, 40, 50, 80]

for cls in landcover_classes:
    df[f"lc_{cls}"] = (df["landcover"] == cls).astype(np.float32)


FEATURES = [
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

X_all = df[FEATURES].astype(np.float32).values
y_all = df["label"].astype(np.int64).values


# ============================================================================
# 2 KM SPATIAL BLOCKS
# ============================================================================

print("\nCreating 2 km spatial blocks...")

df["_block_x"] = np.floor(df["x"] / BLOCK_SIZE).astype(int)
df["_block_y"] = np.floor(df["y"] / BLOCK_SIZE).astype(int)
groups = (
    df["_block_x"].astype(str)
    + "_"
    + df["_block_y"].astype(str)
).values

print("Spatial blocks:", len(np.unique(groups)))


# ============================================================================
# OUTER SPATIAL TEST SPLIT
# ============================================================================

print("\nCreating spatial train/test split...")

outer = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED,
)

train_idx, test_idx = next(
    outer.split(X_all, y_all, groups)
)

print("Training pool:", len(train_idx))
print("Test set:", len(test_idx))


# ============================================================================
# INNER SPATIAL VALIDATION SPLIT
# ============================================================================

print("\nCreating validation split...")

inner = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED,
)

inner_groups = groups[train_idx]

inner_train_rel, val_rel = next(
    inner.split(
        X_all[train_idx],
        y_all[train_idx],
        inner_groups,
    )
)

inner_train_idx = train_idx[inner_train_rel]
val_idx = train_idx[val_rel]

print("Training samples:", len(inner_train_idx))
print("Validation samples:", len(val_idx))
print("Testing samples:", len(test_idx))

print("\nTraining distribution:")
print(pd.Series(y_all[inner_train_idx]).value_counts())

print("\nValidation distribution:")
print(pd.Series(y_all[val_idx]).value_counts())

print("\nTesting distribution:")
print(pd.Series(y_all[test_idx]).value_counts())


# ============================================================================
# NORMALIZATION
# ============================================================================

# Fit scaler ONLY on the training data.
scaler = StandardScaler()

X_train = scaler.fit_transform(X_all[inner_train_idx])
X_val = scaler.transform(X_all[val_idx])
X_test = scaler.transform(X_all[test_idx])

y_train = y_all[inner_train_idx]
y_val = y_all[val_idx]
y_test = y_all[test_idx]


# ============================================================================
# DEEP RESIDUAL MLP
# ============================================================================

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


def make_loader(X, y, shuffle):
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
    )


train_loader = make_loader(X_train, y_train, True)
val_loader = make_loader(X_val, y_val, False)
test_loader = make_loader(X_test, y_test, False)


# ============================================================================
# EVALUATION
# ============================================================================

def predict_model(model, loader):
    model.eval()

    probabilities = []
    labels = []

    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb)
            probs = torch.sigmoid(logits)

            probabilities.extend(probs.cpu().numpy())
            labels.extend(yb.cpu().numpy())

    return np.asarray(labels), np.asarray(probabilities)


def evaluate(model, loader):
    labels, probabilities = predict_model(model, loader)

    predictions = (probabilities >= 0.5).astype(int)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(
            labels, predictions, zero_division=0
        ),
        "recall": recall_score(
            labels, predictions, zero_division=0
        ),
        "f1": f1_score(
            labels, predictions, zero_division=0
        ),
        "auc": roc_auc_score(labels, probabilities),
        "labels": labels,
        "probabilities": probabilities,
        "predictions": predictions,
    }


# ============================================================================
# TRAINING
# ============================================================================

def train_one_model(model_seed):
    seed_everything(model_seed)

    model = DeepResMLP(len(FEATURES)).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    criterion = nn.BCEWithLogitsLoss()

    best_auc = -np.inf
    best_state = None
    patience_counter = 0

    print("\n" + "-" * 70)
    print(f"TRAINING MODEL {model_seed}")
    print("-" * 70)

    for epoch in range(1, EPOCHS + 1):
        model.train()

        running_loss = 0.0
        n = 0

        for xb, yb in train_loader:
            optimizer.zero_grad()

            logits = model(xb)
            loss = criterion(logits, yb)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=2.0,
            )

            optimizer.step()

            running_loss += loss.item() * len(xb)
            n += len(xb)

        train_loss = running_loss / n

        val_metrics = evaluate(model, val_loader)

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f} | "
            f"Val AUC: {val_metrics['auc']:.4f}"
        )

        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

            patience_counter = 0
            print(
                f"  -> Best validation AUC: {best_auc:.4f}"
            )
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print("Early stopping.")
            break

    model.load_state_dict(best_state)

    return model, best_auc


# ============================================================================
# DEEP ENSEMBLE
# ============================================================================

print("\n" + "=" * 70)
print("STARTING DEEP RESMLP ENSEMBLE")
print("=" * 70)

models = []
validation_probabilities = []
test_probabilities = []

for i in range(N_MODELS):
    model_seed = SEED + i * 100

    model, best_val_auc = train_one_model(model_seed)

    val_labels, val_probs = predict_model(model, val_loader)
    test_labels, test_probs = predict_model(model, test_loader)

    models.append(model)
    validation_probabilities.append(val_probs)
    test_probabilities.append(test_probs)

    print(
        f"Model {i + 1} best validation AUC: "
        f"{best_val_auc:.4f}"
    )


# Average independently trained deep models.
val_ensemble = np.mean(
    np.vstack(validation_probabilities),
    axis=0,
)

test_ensemble = np.mean(
    np.vstack(test_probabilities),
    axis=0,
)


# ============================================================================
# VALIDATION-ONLY THRESHOLD SELECTION
# ============================================================================

# Find the classification threshold using ONLY validation data.
# AUC itself is threshold-independent; threshold selection only affects
# accuracy/precision/recall/F1.
best_threshold = 0.50
best_val_f1 = -1.0

for threshold in np.arange(0.20, 0.81, 0.01):
    val_pred = (val_ensemble >= threshold).astype(int)

    score = f1_score(
        y_val,
        val_pred,
        zero_division=0,
    )

    if score > best_val_f1:
        best_val_f1 = score
        best_threshold = float(threshold)


# ============================================================================
# FINAL TEST EVALUATION
# ============================================================================

test_pred = (
    test_ensemble >= best_threshold
).astype(int)

test_auc = roc_auc_score(y_test, test_ensemble)
test_accuracy = accuracy_score(y_test, test_pred)
test_precision = precision_score(
    y_test,
    test_pred,
    zero_division=0,
)
test_recall = recall_score(
    y_test,
    test_pred,
    zero_division=0,
)
test_f1 = f1_score(
    y_test,
    test_pred,
    zero_division=0,
)

cm = confusion_matrix(y_test, test_pred)

print("\n" + "=" * 70)
print("FINAL DEEP RESMLP STATIC RESULTS")
print("=" * 70)

print(f"Validation-selected threshold: {best_threshold:.2f}")
print(f"Validation F1 at threshold:   {best_val_f1:.4f}")

print()
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
        y_test,
        test_pred,
        target_names=[
            "Non-Landslide",
            "Landslide",
        ],
        digits=4,
    )
)


# ============================================================================
# SAVE MODEL
# ============================================================================

# Save the ensemble weights and scaler parameters needed for inference.
torch.save(
    {
        "models": [
            model.state_dict()
            for model in models
        ],
        "input_features": FEATURES,
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "threshold": best_threshold,
        "seed": SEED,
    },
    MODEL_PATH,
)

metrics_text = f"""
DEEP RESMLP STATIC LANDSLIDE SUSCEPTIBILITY

Dataset:
{DATASET_PATH}

Features:
{FEATURES}

Samples:
Training   : {len(y_train)}
Validation : {len(y_val)}
Testing    : {len(y_test)}

Models in ensemble:
{N_MODELS}

Validation-selected threshold:
{best_threshold:.4f}

Validation F1:
{best_val_f1:.4f}

FINAL SPATIAL TEST RESULTS
Accuracy : {test_accuracy:.4f}
Precision: {test_precision:.4f}
Recall   : {test_recall:.4f}
F1 Score : {test_f1:.4f}
ROC-AUC  : {test_auc:.4f}

Confusion Matrix:
{cm}
"""

METRICS_PATH.write_text(
    metrics_text.strip() + "\n",
    encoding="utf-8",
)

print("\nModel saved to:")
print(MODEL_PATH)

print("\nMetrics saved to:")
print(METRICS_PATH)

print("\n" + "=" * 70)
print("DEEP RESMLP TRAINING COMPLETE")
print("=" * 70)
