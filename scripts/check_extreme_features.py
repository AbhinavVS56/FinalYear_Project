import pandas as pd

DATASET = "../datasets/processed/training/idukki_training_dataset.csv"

df = pd.read_csv(DATASET)

print("================================")
print("EXTREME FEATURE CHECK")
print("================================")


# --------------------------------------------------
# TRI
# --------------------------------------------------

print("\nTRI > 20")
print("--------------------------------")

tri_extreme = df[df["tri"] > 20]

print("Count:", len(tri_extreme))

print(tri_extreme.groupby("label").size())

print("\nTop TRI values:")
print(
    tri_extreme
    .sort_values("tri", ascending=False)
    .head(20)
)


# --------------------------------------------------
# CURVATURE
# --------------------------------------------------

print("\n\nCURVATURE |value| > 0.05")
print("--------------------------------")

curv_extreme = df[df["curvature"].abs() > 0.05]

print("Count:", len(curv_extreme))

print(curv_extreme.groupby("label").size())

print("\nExtreme curvature values:")
print(
    curv_extreme
    .sort_values("curvature")
    .head(20)
)


# --------------------------------------------------
# SLOPE
# --------------------------------------------------

print("\n\nSLOPE > 60°")
print("--------------------------------")

slope_extreme = df[df["slope"] > 60]

print("Count:", len(slope_extreme))

print(slope_extreme.groupby("label").size())


# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

print("\n================================")
print("DONE")
print("================================")