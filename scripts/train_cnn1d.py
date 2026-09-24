import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)

# ============================================================
# 1D CNN FOR IDUKKI STATIC LANDSLIDE SUSCEPTIBILITY
# Same dataset + same 2 km spatial test protocol as train_deep.py
# ============================================================

DATA_PATH = "../datasets/processed/training/idukki_training_dataset_v2.csv"
MODEL_DIR = "../models"
RESULT_DIR = "../results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

SEED_LIST = [42, 142, 242]
DEVICE = torch.device("cpu")

EPOCHS = 200
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 25


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# Load data
# ============================================================

df = pd.read_csv(DATA_PATH)

print("=" * 70)
print("1D CNN STATIC LANDSLIDE SUSCEPTIBILITY")
print("=" * 70)
print(f"Dataset shape: {df.shape}")

# ------------------------------------------------------------
# Build exactly the same 13 static features used by Deep ResMLP
# ------------------------------------------------------------

df["aspect_rad"] = np.deg2rad(df["aspect"])
df["aspect_sin"] = np.sin(df["aspect_rad"])
df["aspect_cos"] = np.cos(df["aspect_rad"])

LANDCOVER_CLASSES = [10, 30, 40, 50, 80]

for cls in LANDCOVER_CLASSES:
    df[f"lc_{cls}"] = (df["landcover"] == cls).astype(float)

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

X = df[FEATURES].astype(np.float32).values
y = df["label"].astype(np.int64).values

print(f"Features: {len(FEATURES)}")
print(FEATURES)
print(f"Class distribution: {np.bincount(y)}")


# ============================================================
# 2 km spatial groups
# Same grouping idea used in the previous experiments
# ============================================================

BLOCK_SIZE = 2000

group_x = np.floor(df["x"].values / BLOCK_SIZE).astype(np.int64)
group_y = np.floor(df["y"].values / BLOCK_SIZE).astype(np.int64)

groups = group_x * 100000 + group_y

outer_cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

train_idx, test_idx = next(
    outer_cv.split(X, y, groups=groups)
)

X_outer_train = X[train_idx]
y_outer_train = y[train_idx]
groups_outer_train = groups[train_idx]

X_test = X[test_idx]
y_test = y[test_idx]

print()
print("OUTER SPATIAL SPLIT")
print(f"Training pool: {len(train_idx)}")
print(f"Test set:      {len(test_idx)}")
print(f"Test positives: {np.sum(y_test == 1)}")
print(f"Test negatives: {np.sum(y_test == 0)}")
print(f"Test blocks: {len(np.unique(groups[test_idx]))}")


# ============================================================
# Inner spatial validation split
# ============================================================

inner_cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=123
)

inner_train_idx, val_idx = next(
    inner_cv.split(
        X_outer_train,
        y_outer_train,
        groups=groups_outer_train
    )
)

X_train = X_outer_train[inner_train_idx]
y_train = y_outer_train[inner_train_idx]

X_val = X_outer_train[val_idx]
y_val = y_outer_train[val_idx]

print()
print("INNER SPATIAL SPLIT")
print(f"Train: {len(X_train)}")
print(f"Val:   {len(X_val)}")
print(f"Test:  {len(X_test)}")


# ============================================================
# Standardization
# IMPORTANT: fit scaler only on training data
# ============================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train).astype(np.float32)
X_val = scaler.transform(X_val).astype(np.float32)
X_test_scaled = scaler.transform(X_test).astype(np.float32)


# ============================================================
# PyTorch datasets
# ============================================================

class TabularDataset(torch.utils.data.Dataset):

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================================================
# 1D CNN
#
# The 13 static factors are represented as a 1D feature sequence.
#
# Input:
#   [batch, 1, 13]
#
# CNN learns local feature interactions through 1D kernels.
# ============================================================

class CNN1D(nn.Module):

    def __init__(self, n_features=13):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv1d(
                in_channels=1,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.15),

            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Conv1d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.25),
        )

        self.global_avg = nn.AdaptiveAvgPool1d(1)
        self.global_max = nn.AdaptiveMaxPool1d(1)

        self.classifier = nn.Sequential(

            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.30),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(64, 1)
        )

    def forward(self, x):

        # [batch, 13] -> [batch, 1, 13]
        x = x.unsqueeze(1)

        x = self.features(x)

        avg = self.global_avg(x).squeeze(-1)
        mx = self.global_max(x).squeeze(-1)

        x = torch.cat([avg, mx], dim=1)

        return self.classifier(x).squeeze(1)


# ============================================================
# Data loaders
# ============================================================

train_loader = torch.utils.data.DataLoader(
    TabularDataset(X_train, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = torch.utils.data.DataLoader(
    TabularDataset(X_val, y_val),
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = torch.utils.data.DataLoader(
    TabularDataset(X_test_scaled, y_test),
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# Training function
# ============================================================

def train_model(seed):

    print()
    print("=" * 70)
    print(f"TRAINING 1D CNN - SEED {seed}")
    print("=" * 70)

    set_seed(seed)

    model = CNN1D(len(FEATURES)).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    criterion = nn.BCEWithLogitsLoss()

    best_auc = -np.inf
    best_state = None
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        model.train()

        train_losses = []

        for xb, yb in train_loader:

            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()

            logits = model(xb)

            loss = criterion(logits, yb)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0
            )

            optimizer.step()

            train_losses.append(loss.item())

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        model.eval()

        val_probs = []
        val_true = []

        with torch.no_grad():

            for xb, yb in val_loader:

                logits = model(xb)

                probs = torch.sigmoid(logits)

                val_probs.extend(probs.cpu().numpy())
                val_true.extend(yb.cpu().numpy())

        val_probs = np.asarray(val_probs)
        val_true = np.asarray(val_true)

        val_auc = roc_auc_score(val_true, val_probs)

        if val_auc > best_auc:

            best_auc = val_auc
            best_epoch = epoch

            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

            patience_counter = 0

        else:

            patience_counter += 1

        if epoch == 1 or epoch % 5 == 0:

            print(
                f"Epoch {epoch:03d} | "
                f"Loss {np.mean(train_losses):.4f} | "
                f"Val AUC {val_auc:.4f} | "
                f"Best {best_auc:.4f}"
            )

        if patience_counter >= PATIENCE:

            print(
                f"Early stopping at epoch {epoch} "
                f"(best epoch {best_epoch})"
            )

            break

    model.load_state_dict(best_state)

    print(
        f"Best validation AUC: {best_auc:.4f} "
        f"at epoch {best_epoch}"
    )

    return model, best_auc


# ============================================================
# Train ensemble
# ============================================================

models = []
val_predictions = []
test_predictions = []

for seed in SEED_LIST:

    model, val_auc = train_model(seed)

    models.append(model)

    # Validation predictions
    model.eval()

    vp = []

    with torch.no_grad():

        for xb, _ in val_loader:

            probs = torch.sigmoid(
                model(xb)
            )

            vp.extend(probs.cpu().numpy())

    val_predictions.append(np.asarray(vp))

    # Test predictions
    tp = []

    with torch.no_grad():

        for xb, _ in test_loader:

            probs = torch.sigmoid(
                model(xb)
            )

            tp.extend(probs.cpu().numpy())

    test_predictions.append(np.asarray(tp))


# ============================================================
# Ensemble
# ============================================================

val_ensemble = np.mean(
    np.vstack(val_predictions),
    axis=0
)

test_ensemble = np.mean(
    np.vstack(test_predictions),
    axis=0
)


# ============================================================
# Choose threshold ONLY using validation set
# ============================================================

thresholds = np.arange(
    0.30,
    0.71,
    0.01
)

best_threshold = 0.50
best_f1 = -1

for threshold in thresholds:

    pred = (val_ensemble >= threshold).astype(int)

    score = f1_score(
        y_val,
        pred,
        zero_division=0
    )

    if score > best_f1:

        best_f1 = score
        best_threshold = float(threshold)


# ============================================================
# Final untouched test evaluation
# ============================================================

test_pred = (
    test_ensemble >= best_threshold
).astype(int)

accuracy = accuracy_score(
    y_test,
    test_pred
)

precision = precision_score(
    y_test,
    test_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    test_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    test_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    test_ensemble
)

cm = confusion_matrix(
    y_test,
    test_pred
)


# ============================================================
# Results
# ============================================================

print()
print("=" * 70)
print("FINAL 1D CNN ENSEMBLE RESULTS")
print("=" * 70)

print(f"Validation threshold: {best_threshold:.2f}")
print(f"Validation F1:        {best_f1:.4f}")
print()
print(f"Test Accuracy:        {accuracy:.4f}")
print(f"Test Precision:       {precision:.4f}")
print(f"Test Recall:          {recall:.4f}")
print(f"Test F1:              {f1:.4f}")
print(f"Test ROC-AUC:         {roc_auc:.4f}")
print()
print("Confusion Matrix:")
print(cm)
print()
print("Classification Report:")
print(
    classification_report(
        y_test,
        test_pred,
        target_names=[
            "Non-landslide",
            "Landslide"
        ],
        digits=4
    )
)


# ============================================================
# Save model + preprocessing information
# ============================================================

checkpoint = {
    "models": [
        model.state_dict()
        for model in models
    ],
    "input_features": FEATURES,
    "scaler_mean": scaler.mean_.tolist(),
    "scaler_scale": scaler.scale_.tolist(),
    "threshold": best_threshold,
    "seeds": SEED_LIST,
    "architecture": "CNN1D"
}

torch.save(
    checkpoint,
    os.path.join(
        MODEL_DIR,
        "cnn1d_static.pt"
    )
)


# ============================================================
# Save metrics
# ============================================================

metrics_path = os.path.join(
    RESULT_DIR,
    "cnn1d_static_metrics.txt"
)

with open(metrics_path, "w") as f:

    f.write("1D CNN STATIC LANDSLIDE SUSCEPTIBILITY\n")
    f.write("=" * 60 + "\n\n")

    f.write(f"Dataset: {DATA_PATH}\n")
    f.write(f"Features: {FEATURES}\n")
    f.write(f"Seeds: {SEED_LIST}\n")
    f.write(f"Test samples: {len(y_test)}\n\n")

    f.write(
        f"Validation threshold: {best_threshold:.4f}\n"
    )

    f.write(
        f"Validation F1: {best_f1:.4f}\n\n"
    )

    f.write(
        f"Test Accuracy: {accuracy:.4f}\n"
    )

    f.write(
        f"Test Precision: {precision:.4f}\n"
    )

    f.write(
        f"Test Recall: {recall:.4f}\n"
    )

    f.write(
        f"Test F1: {f1:.4f}\n"
    )

    f.write(
        f"Test ROC-AUC: {roc_auc:.4f}\n\n"
    )

    f.write("Confusion Matrix:\n")
    f.write(str(cm))
    f.write("\n\n")

    f.write("Classification Report:\n")
    f.write(
        classification_report(
            y_test,
            test_pred,
            target_names=[
                "Non-landslide",
                "Landslide"
            ],
            digits=4
        )
    )

print()
print("=" * 70)
print("SAVED")
print("=" * 70)
print("Model:")
print("models/cnn1d_static.pt")
print()
print("Metrics:")
print("results/cnn1d_static_metrics.txt")
print("=" * 70)
