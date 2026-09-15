import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

POSITIVE_PATH = (
    PROJECT_ROOT / "datasets" / "processed" /
    "training" / "idukki_positive_samples.csv"
)

NEGATIVE_PATH = (
    PROJECT_ROOT / "datasets" / "processed" /
    "training" / "idukki_negative_samples.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT / "datasets" / "processed" /
    "training" / "idukki_training_dataset.csv"
)


print("Loading positive samples...")
positive = pd.read_csv(POSITIVE_PATH)

print("Loading negative samples...")
negative = pd.read_csv(NEGATIVE_PATH)


print("\nCombining datasets...")

df = pd.concat(
    [positive, negative],
    ignore_index=True
)


# Shuffle while preserving coordinates
df = df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# Remove invalid values
df = df.replace(
    [float("inf"), float("-inf")],
    pd.NA
)

df = df.dropna()


# Save
df.to_csv(
    OUTPUT_PATH,
    index=False
)


print("\n--------------------------------")
print("TRAINING DATASET CREATED")
print("--------------------------------")

print("Total samples:", len(df))

print("\nColumns:")
print(df.columns.tolist())

print("\nClass distribution:")
print(df["label"].value_counts())

print("\nDataset shape:")
print(df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nSaved to:")
print(OUTPUT_PATH)