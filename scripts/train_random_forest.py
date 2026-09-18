import pandas as pd
import numpy as np
from pathlib import Path

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

import joblib


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET = PROJECT_ROOT / "datasets" / "processed" / "training" / "idukki_training_dataset.csv"

MODEL_DIR = PROJECT_ROOT / "models"
RESULT_DIR = PROJECT_ROOT / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(DATASET)

print("Dataset shape:", df.shape)
print("Columns:", list(df.columns))


# ============================================================
# 2. CONVERT ASPECT TO SIN/COS
# ============================================================

print("\nConverting aspect to sin/cos...")

angle = np.radians(df["aspect"])

df["aspect_sin"] = np.sin(angle)
df["aspect_cos"] = np.cos(angle)


# ============================================================
# 3. CREATE MODEL FEATURES
# ============================================================

features = [
    "elevation",
    "slope",
    "aspect_sin",
    "aspect_cos",
    "curvature",
    "tri"
]

X = df[features]
y = df["label"]


# ============================================================
# 4. CREATE SPATIAL BLOCKS
# ============================================================

print("\nCreating spatial blocks...")

# 2 km spatial blocks
BLOCK_SIZE = 2000

df["block_x"] = np.floor(df["x"] / BLOCK_SIZE).astype(int)
df["block_y"] = np.floor(df["y"] / BLOCK_SIZE).astype(int)

# Unique ID for each spatial block
groups = (
    df["block_x"].astype(str)
    + "_"
    + df["block_y"].astype(str)
)


print("Number of spatial blocks:", groups.nunique())


# ============================================================
# 5. SPATIAL TRAIN / TEST SPLIT
# ============================================================

print("\nCreating spatial train/test split...")

sgkf = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

train_idx, test_idx = next(
    sgkf.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())


# ============================================================
# 6. TRAIN RANDOM FOREST
# ============================================================

print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

model.fit(X_train, y_train)

print("Training complete.")


# ============================================================
# 7. PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:, 1]


# ============================================================
# 8. EVALUATION
# ============================================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
auc = roc_auc_score(y_test, y_prob)

cm = confusion_matrix(y_test, y_pred)


print("\n")
print("=" * 60)
print("RANDOM FOREST RESULTS")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {auc:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Non-Landslide", "Landslide"],
        zero_division=0
    )
)


# ============================================================
# 9. FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print("\nFeature Importance:")
print(importance.to_string(index=False))


# ============================================================
# 10. SAVE MODEL
# ============================================================

model_path = MODEL_DIR / "random_forest_static.joblib"

joblib.dump(model, model_path)

print("\nModel saved to:")
print(model_path)


# ============================================================
# 11. SAVE FEATURE IMPORTANCE
# ============================================================

importance_path = RESULT_DIR / "random_forest_feature_importance.csv"

importance.to_csv(
    importance_path,
    index=False
)

print("Feature importance saved to:")
print(importance_path)


# ============================================================
# 12. SAVE METRICS
# ============================================================

metrics_path = RESULT_DIR / "random_forest_metrics.txt"

with open(metrics_path, "w") as f:

    f.write("RANDOM FOREST STATIC SUSCEPTIBILITY MODEL\n")
    f.write("=" * 50 + "\n\n")

    f.write(f"Dataset size: {len(df)}\n")
    f.write(f"Training samples: {len(X_train)}\n")
    f.write(f"Testing samples: {len(X_test)}\n")
    f.write(f"Spatial block size: {BLOCK_SIZE} m\n\n")

    f.write(f"Accuracy : {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall   : {recall:.4f}\n")
    f.write(f"F1 Score : {f1:.4f}\n")
    f.write(f"ROC-AUC  : {auc:.4f}\n\n")

    f.write("Confusion Matrix:\n")
    f.write(str(cm))
    f.write("\n\n")

    f.write("Feature Importance:\n")
    f.write(importance.to_string(index=False))

print("Metrics saved to:")
print(metrics_path)


# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("=" * 60)
print("STATIC RANDOM FOREST PIPELINE COMPLETE")
print("=" * 60)