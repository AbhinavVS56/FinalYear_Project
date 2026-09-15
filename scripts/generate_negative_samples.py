import geopandas as gpd
import rasterio
import numpy as np
import pandas as pd

from rasterio.features import rasterize
from rasterio.mask import mask
from scipy.ndimage import distance_transform_edt


# --------------------------------------------------
# PATHS
# --------------------------------------------------

LANDSLIDE_FILE = "../datasets/processed/landslides/idukki_landslides.gpkg"

BOUNDARY_FILE = "../datasets/raw/boundaries/idukki_boundary.gpkg"

FEATURE_DIR = "../datasets/processed/terrain_features"

OUTPUT_FILE = "../datasets/processed/training/idukki_negative_samples.csv"


# --------------------------------------------------
# LOAD LANDSLIDES
# --------------------------------------------------

print("Loading landslide inventory...")

landslides = gpd.read_file(LANDSLIDE_FILE)

print(f"Landslide points: {len(landslides)}")


# --------------------------------------------------
# LOAD IDUKKI BOUNDARY
# --------------------------------------------------

print("\nLoading Idukki boundary...")

boundary = gpd.read_file(BOUNDARY_FILE)

print(f"Boundary CRS: {boundary.crs}")


# --------------------------------------------------
# LOAD ELEVATION
# --------------------------------------------------

print("\nLoading elevation raster...")

elevation_file = f"{FEATURE_DIR}/idukki_elevation.tif"

with rasterio.open(elevation_file) as src:

    elevation = src.read(1)

    transform = src.transform
    crs = src.crs

    height = src.height
    width = src.width

print(f"Raster size: {height} x {width}")
print(f"CRS: {crs}")


# --------------------------------------------------
# REPROJECT BOUNDARY
# --------------------------------------------------

boundary = boundary.to_crs(crs)


# --------------------------------------------------
# CREATE IDUKKI MASK
# --------------------------------------------------

print("\nCreating Idukki boundary mask...")

idukki_mask = rasterize(
    [(geom, 1) for geom in boundary.geometry],
    out_shape=(height, width),
    transform=transform,
    fill=0,
    dtype="uint8"
)

print(f"Idukki pixels: {np.sum(idukki_mask == 1)}")


# --------------------------------------------------
# CREATE LANDSLIDE MASK
# --------------------------------------------------

print("\nCreating landslide mask...")

landslides = landslides.to_crs(crs)

landslide_mask = rasterize(
    [(geom, 1) for geom in landslides.geometry],
    out_shape=(height, width),
    transform=transform,
    fill=0,
    dtype="uint8"
)

print(f"Landslide pixels: {np.sum(landslide_mask == 1)}")


# --------------------------------------------------
# CREATE 200 m EXCLUSION ZONE
# --------------------------------------------------

print("\nCreating 200 m exclusion zone...")

pixel_size = abs(transform.a)

distance = distance_transform_edt(
    landslide_mask == 0
) * pixel_size

exclusion_zone = distance < 200

print(
    f"Excluded pixels: "
    f"{np.sum(exclusion_zone & (idukki_mask == 1))}"
)


# --------------------------------------------------
# VALID TERRAIN PIXELS
# --------------------------------------------------

print("\nFinding valid candidate pixels...")

valid_elevation = (
    np.isfinite(elevation) &
    (elevation > 0)
)

candidate_mask = (
    (idukki_mask == 1) &
    valid_elevation &
    (landslide_mask == 0) &
    (~exclusion_zone)
)

rows, cols = np.where(candidate_mask)

print(f"Candidate negative pixels: {len(rows)}")


# --------------------------------------------------
# CHECK CANDIDATES
# --------------------------------------------------

if len(rows) < len(landslides):

    raise ValueError(
        "Not enough valid pixels available for "
        "negative sampling."
    )


# --------------------------------------------------
# SPATIAL DISTRIBUTION
# --------------------------------------------------

print("\nSelecting spatially distributed negative samples...")

rng = np.random.default_rng(42)

target = len(landslides)

# Divide Idukki into spatial blocks
block_rows = 20
block_cols = 20

r_block = np.clip(
    (rows / height * block_rows).astype(int),
    0,
    block_rows - 1
)

c_block = np.clip(
    (cols / width * block_cols).astype(int),
    0,
    block_cols - 1
)

blocks = {}

for i, (r, c) in enumerate(zip(r_block, c_block)):

    key = (r, c)

    if key not in blocks:
        blocks[key] = []

    blocks[key].append(i)


# Shuffle blocks
block_keys = list(blocks.keys())
rng.shuffle(block_keys)

selected = []


# --------------------------------------------------
# FIRST PASS
# Take samples from different spatial blocks
# --------------------------------------------------

for key in block_keys:

    indices = blocks[key]

    if len(indices) > 0:

        selected.append(
            rng.choice(indices)
        )

    if len(selected) >= target:
        break


# --------------------------------------------------
# SECOND PASS
# Fill remaining samples
# --------------------------------------------------

if len(selected) < target:

    selected = np.array(selected)

    remaining = np.setdiff1d(
        np.arange(len(rows)),
        selected
    )

    extra = rng.choice(
        remaining,
        size=target - len(selected),
        replace=False
    )

    selected = np.concatenate(
        [selected, extra]
    )

else:

    selected = np.array(selected)


selected = selected[:target]


rows_selected = rows[selected]
cols_selected = cols[selected]


print(
    f"Negative samples selected: "
    f"{len(rows_selected)}"
)


# --------------------------------------------------
# EXTRACT TERRAIN FEATURES
# --------------------------------------------------

print("\nExtracting terrain features...")

features = {}

feature_names = [
    "elevation",
    "slope",
    "aspect",
    "curvature",
    "tri"
]


for name in feature_names:

    path = f"{FEATURE_DIR}/idukki_{name}.tif"

    with rasterio.open(path) as src:

        data = src.read(1)

        features[name] = data[
            rows_selected,
            cols_selected
        ]


# --------------------------------------------------
# CREATE DATAFRAME
# --------------------------------------------------

df = pd.DataFrame(features)

df["label"] = 0


# --------------------------------------------------
# REMOVE INVALID VALUES
# --------------------------------------------------

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

df = df.dropna()


# --------------------------------------------------
# SAVE
# --------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# --------------------------------------------------
# FINAL REPORT
# --------------------------------------------------

print("\n--------------------------------")
print("NEGATIVE SAMPLE GENERATION DONE")
print("--------------------------------")

print(f"Samples saved: {len(df)}")

print(f"Output: {OUTPUT_FILE}")

print("\nFeature summary:")

print(df.describe())

print("\nLabel distribution:")

print(df["label"].value_counts())