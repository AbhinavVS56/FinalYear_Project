import pandas as pd
import numpy as np

# --------------------------------------------------
# PATH
# --------------------------------------------------

DATASET = "../datasets/processed/training/idukki_training_dataset.csv"


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

print("Loading training dataset...")

df = pd.read_csv(DATASET)

print("\n================================")
print("DATASET QUALITY CHECK")
print("================================")


# --------------------------------------------------
# BASIC INFORMATION
# --------------------------------------------------

print("\n1. BASIC INFORMATION")
print("--------------------------------")

print("Shape:", df.shape)

print("Columns:")
print(df.columns.tolist())


# --------------------------------------------------
# MISSING VALUES
# --------------------------------------------------

print("\n2. MISSING VALUES")
print("--------------------------------")

print(df.isnull().sum())

print(
    "Total missing values:",
    df.isnull().sum().sum()
)


# --------------------------------------------------
# INFINITE VALUES
# --------------------------------------------------

print("\n3. INFINITE VALUES")
print("--------------------------------")

numeric = df.select_dtypes(include=np.number)

print(
    "Infinite values:",
    np.isinf(numeric).sum().sum()
)


# --------------------------------------------------
# DUPLICATES
# --------------------------------------------------

print("\n4. DUPLICATE ROWS")
print("--------------------------------")

print(
    "Duplicate rows:",
    df.duplicated().sum()
)

print(
    "Duplicate feature rows:",
    df.drop(columns=["label"]).duplicated().sum()
)


# --------------------------------------------------
# CLASS BALANCE
# --------------------------------------------------

print("\n5. CLASS DISTRIBUTION")
print("--------------------------------")

print(df["label"].value_counts())

print("\nPercentages:")

print(
    df["label"].value_counts(normalize=True) * 100
)


# --------------------------------------------------
# FEATURE STATISTICS
# --------------------------------------------------

features = [
    "elevation",
    "slope",
    "aspect",
    "curvature",
    "tri"
]

print("\n6. FEATURE STATISTICS")
print("--------------------------------")

print(
    df[features].describe().T
)


# --------------------------------------------------
# ZERO VALUES
# --------------------------------------------------

print("\n7. ZERO VALUES")
print("--------------------------------")

for feature in features:

    count = (df[feature] == 0).sum()

    print(
        f"{feature:12s}: {count:5d} "
        f"({count / len(df) * 100:.2f}%)"
    )


# --------------------------------------------------
# EXTREME VALUES
# --------------------------------------------------

print("\n8. EXTREME VALUES")
print("--------------------------------")

for feature in features:

    q1 = df[feature].quantile(0.25)
    q3 = df[feature].quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = (
        (df[feature] < lower) |
        (df[feature] > upper)
    ).sum()

    print(
        f"{feature:12s}: "
        f"{outliers} potential outliers"
    )


# --------------------------------------------------
# CLASS-WISE STATISTICS
# --------------------------------------------------

print("\n9. CLASS-WISE MEANS")
print("--------------------------------")

print(
    df.groupby("label")[features].mean()
)


# --------------------------------------------------
# CLASS-WISE MEDIANS
# --------------------------------------------------

print("\n10. CLASS-WISE MEDIANS")
print("--------------------------------")

print(
    df.groupby("label")[features].median()
)


# --------------------------------------------------
# CORRELATION
# --------------------------------------------------

print("\n11. FEATURE CORRELATION")
print("--------------------------------")

print(
    df[features].corr().round(3)
)


# --------------------------------------------------
# ASPECT CHECK
# --------------------------------------------------

print("\n12. ASPECT CHECK")
print("--------------------------------")

print(
    "Aspect minimum:",
    df["aspect"].min()
)

print(
    "Aspect maximum:",
    df["aspect"].max()
)

print(
    "Aspect near 0°:",
    (df["aspect"] < 5).sum()
)

print(
    "Aspect near 360°:",
    (df["aspect"] > 355).sum()
)


# --------------------------------------------------
# FINAL CHECKS
# --------------------------------------------------

print("\n================================")
print("FINAL QUALITY SUMMARY")
print("================================")

checks = {
    "Correct number of rows": len(df) == 4434,
    "Correct number of columns": len(df.columns) == 6,
    "No missing values": df.isnull().sum().sum() == 0,
    "No infinite values": np.isinf(numeric).sum().sum() == 0,
    "Balanced classes": (
        df["label"].value_counts().get(0, 0) == 2217
        and
        df["label"].value_counts().get(1, 0) == 2217
    )
}

for name, result in checks.items():

    status = "PASS" if result else "CHECK"

    print(f"{status:6s} - {name}")

print("\nQuality check completed.")