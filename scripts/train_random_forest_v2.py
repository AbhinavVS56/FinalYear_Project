import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATASET_PATH = os.path.join(
    PROJECT_ROOT,
    "datasets",
    "processed",
    "training",
    "idukki_training_dataset_v2.csv"
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models"
)

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "results"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("RANDOM FOREST - MODEL 2")
print("Terrain + Soil + Land Cover")
print("=" * 60)

df = pd.read_csv(DATASET_PATH)

print(f"\nDataset shape: {df.shape}")


# ============================================================
# CHECK DATA
# ============================================================

print("\nClass distribution:")

print(
    df["label"].value_counts()
)


# ============================================================
# ASPECT -> SIN/COS
# ============================================================

df["aspect_rad"] = np.deg2rad(
    df["aspect"]
)

df["aspect_sin"] = np.sin(
    df["aspect_rad"]
)

df["aspect_cos"] = np.cos(
    df["aspect_rad"]
)


# ============================================================
# LAND COVER ONE-HOT ENCODING
# ============================================================

print("\nLand cover classes:")

print(
    sorted(
        df["landcover"]
        .dropna()
        .unique()
    )
)

# Convert land-cover codes to categorical
# and create one-hot columns.

landcover_dummies = pd.get_dummies(
    df["landcover"],
    prefix="lc",
    dtype=int
)

df = pd.concat(
    [df, landcover_dummies],
    axis=1
)

print(
    "\nCreated land-cover features:"
)

print(
    list(landcover_dummies.columns)
)


# ============================================================
# FEATURES
# ============================================================

base_features = [
    "elevation",
    "slope",
    "aspect_sin",
    "aspect_cos",
    "curvature",
    "tri",
    "clay",
    "sand"
]

landcover_features = list(
    landcover_dummies.columns
)

features = (
    base_features +
    landcover_features
)

X = df[features]

y = df["label"]


# ============================================================
# SPATIAL BLOCKS
# ============================================================

# x/y are NOT used as ML features.
# They are used only to create spatial blocks.

BLOCK_SIZE = 2000  # metres

df["block_x"] = (
    np.floor(
        df["x"] / BLOCK_SIZE
    ).astype(int)
)

df["block_y"] = (
    np.floor(
        df["y"] / BLOCK_SIZE
    ).astype(int)
)

df["spatial_block"] = (
    df["block_x"].astype(str)
    + "_"
    + df["block_y"].astype(str)
)


print(
    f"\nNumber of spatial blocks: "
    f"{df['spatial_block'].nunique()}"
)


# ============================================================
# SPATIAL TRAIN / TEST SPLIT
# ============================================================

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

groups = df["spatial_block"]

# Use the same deterministic first fold
# methodology as Model 1.

train_idx, test_idx = next(
    cv.split(
        X,
        y,
        groups
    )
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

print(
    "\nTraining class distribution:"
)

print(
    y_train.value_counts()
)

print(
    "\nTesting class distribution:"
)

print(
    y_test.value_counts()
)


# ============================================================
# RANDOM FOREST
# ============================================================

print(
    "\nTraining Random Forest..."
)

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

rf.fit(
    X_train,
    y_train
)


# ============================================================
# PREDICTION
# ============================================================

y_pred = rf.predict(
    X_test
)

y_prob = rf.predict_proba(
    X_test
)[:, 1]


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

roc_auc = roc_auc_score(
    y_test,
    y_prob
)

cm = confusion_matrix(
    y_test,
    y_pred
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 60)
print("MODEL 2 RESULTS")
print("=" * 60)

print(
    f"\nAccuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1 Score : {f1:.4f}"
)

print(
    f"ROC-AUC  : {roc_auc:.4f}"
)

print(
    "\nConfusion Matrix:"
)

print(cm)

print(
    "\nClassification Report:"
)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "non-landslide",
            "landslide"
        ]
    )
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "feature": features,
    "importance": rf.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print(
    "\nFeature Importance:"
)

print(
    importance.to_string(
        index=False
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

import joblib

model_path = os.path.join(
    MODEL_DIR,
    "random_forest_static_v2.joblib"
)

joblib.dump(
    rf,
    model_path
)


# ============================================================
# SAVE FEATURE IMPORTANCE
# ============================================================

importance_path = os.path.join(
    RESULTS_DIR,
    "random_forest_v2_feature_importance.csv"
)

importance.to_csv(
    importance_path,
    index=False
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics_path = os.path.join(
    RESULTS_DIR,
    "random_forest_v2_metrics.txt"
)

with open(
    metrics_path,
    "w"
) as f:

    f.write(
        "Random Forest Model 2\n"
    )

    f.write(
        "Terrain + Soil + Land Cover\n\n"
    )

    f.write(
        f"Dataset size: {len(df)}\n"
    )

    f.write(
        f"Training samples: {len(X_train)}\n"
    )

    f.write(
        f"Testing samples: {len(X_test)}\n\n"
    )

    f.write(
        f"Accuracy: {accuracy:.4f}\n"
    )

    f.write(
        f"Precision: {precision:.4f}\n"
    )

    f.write(
        f"Recall: {recall:.4f}\n"
    )

    f.write(
        f"F1 Score: {f1:.4f}\n"
    )

    f.write(
        f"ROC-AUC: {roc_auc:.4f}\n\n"
    )

    f.write(
        "Confusion Matrix:\n"
    )

    f.write(
        str(cm)
    )

    f.write(
        "\n\nClassification Report:\n"
    )

    f.write(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "non-landslide",
                "landslide"
            ]
        )
    )


# ============================================================
# DONE
# ============================================================

print(
    "\nModel saved to:"
)

print(
    model_path
)

print(
    "\nFeature importance saved to:"
)

print(
    importance_path
)

print(
    "\nMetrics saved to:"
)

print(
    metrics_path
)

print(
    "\n" + "=" * 60
)

print(
    "MODEL 2 COMPLETE"
)

print(
    "=" * 60
)