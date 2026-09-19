import os
import joblib
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = "models/random_forest_static_v2.joblib"

OUTPUT_DIR = "datasets/processed/susceptibility"
OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "idukki_static_susceptibility.tif"
)

RASTERS = {
    "elevation": "datasets/processed/terrain_features/idukki_elevation.tif",
    "slope": "datasets/processed/terrain_features/idukki_slope.tif",
    "aspect": "datasets/processed/terrain_features/idukki_aspect.tif",
    "curvature": "datasets/processed/terrain_features/idukki_curvature.tif",
    "tri": "datasets/processed/terrain_features/idukki_tri.tif",
    "clay": "datasets/processed/soil/idukki_clay_0_5cm.tif",
    "sand": "datasets/processed/soil/idukki_sand_0_5cm.tif",
    "landcover": "datasets/processed/landcover/idukki_worldcover2021.tif"
}


# ============================================================
# MODEL FEATURES
# ============================================================

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
    "lc_80"
]


# ============================================================
# BLOCK SIZE
# ============================================================

# Small enough for an 8 GB RAM computer
BLOCK_ROWS = 16


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading Random Forest model...")

model = joblib.load(MODEL_PATH)

print("Model loaded.")

print("Model features:")
print(list(model.feature_names_in_))

if list(model.feature_names_in_) != FEATURES:
    raise ValueError(
        "Model feature names/order do not match expected features."
    )

print("Feature order verified.")


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# OPEN ALL RASTERS
# ============================================================

print("\nOpening raster datasets...")

sources = {}

for name, path in RASTERS.items():

    print(f"  {name}: {path}")

    sources[name] = rasterio.open(path)


# ============================================================
# VERIFY RASTER ALIGNMENT
# ============================================================

reference = sources["elevation"]

print("\nReference raster:")
print(f"CRS       : {reference.crs}")
print(f"Width     : {reference.width}")
print(f"Height    : {reference.height}")
print(f"Resolution: {reference.res}")


for name, src in sources.items():

    if src.width != reference.width:
        raise ValueError(
            f"{name} width does not match elevation raster."
        )

    if src.height != reference.height:
        raise ValueError(
            f"{name} height does not match elevation raster."
        )

    if src.transform != reference.transform:
        raise ValueError(
            f"{name} transform does not match elevation raster."
        )

    if src.crs != reference.crs:
        raise ValueError(
            f"{name} CRS does not match elevation raster."
        )


print("\nAll rasters are aligned.")


# ============================================================
# GET CLASS 1 INDEX
# ============================================================

class_1_index = list(model.classes_).index(1)

print("\nModel classes:", model.classes_)
print("Class 1 probability index:", class_1_index)


# ============================================================
# OUTPUT PROFILE
# ============================================================

profile = reference.profile.copy()

profile.update(
    dtype="float32",
    count=1,
    nodata=-9999.0,
    compress="deflate",
    predictor=2
)


# ============================================================
# CREATE OUTPUT RASTER
# ============================================================

print("\nCreating output raster...")

with rasterio.open(OUTPUT_PATH, "w", **profile) as dst:

    total_blocks = int(
        np.ceil(reference.height / BLOCK_ROWS)
    )

    block_number = 0

    # --------------------------------------------------------
    # PROCESS RASTER IN SMALL BLOCKS
    # --------------------------------------------------------

    for row_start in range(
        0,
        reference.height,
        BLOCK_ROWS
    ):

        block_number += 1

        rows = min(
            BLOCK_ROWS,
            reference.height - row_start
        )

        window = Window(
            col_off=0,
            row_off=row_start,
            width=reference.width,
            height=rows
        )

        # ----------------------------------------------------
        # READ RASTER VALUES
        # ----------------------------------------------------

        elevation = sources["elevation"].read(
            1,
            window=window
        ).astype(np.float32)

        slope = sources["slope"].read(
            1,
            window=window
        ).astype(np.float32)

        aspect = sources["aspect"].read(
            1,
            window=window
        ).astype(np.float32)

        curvature = sources["curvature"].read(
            1,
            window=window
        ).astype(np.float32)

        tri = sources["tri"].read(
            1,
            window=window
        ).astype(np.float32)

        clay = sources["clay"].read(
            1,
            window=window
        ).astype(np.float32)

        sand = sources["sand"].read(
            1,
            window=window
        ).astype(np.float32)

        landcover = sources["landcover"].read(
            1,
            window=window
        ).astype(np.int16)

        # ----------------------------------------------------
        # VALID PIXEL MASK
        # ----------------------------------------------------

        valid = np.ones(
            elevation.shape,
            dtype=bool
        )

        for name, array in [
            ("elevation", elevation),
            ("slope", slope),
            ("aspect", aspect),
            ("curvature", curvature),
            ("tri", tri),
            ("clay", clay),
            ("sand", sand)
        ]:

            nodata = sources[name].nodata

            if nodata is not None:
                valid &= array != nodata

            valid &= np.isfinite(array)

        # Only landcover classes represented during training
        supported_landcover = np.isin(
            landcover,
            [10, 30, 40, 50, 80]
        )

        valid &= supported_landcover

        # ----------------------------------------------------
        # OUTPUT BLOCK INITIALIZED AS NODATA
        # ----------------------------------------------------

        output = np.full(
            elevation.shape,
            -9999.0,
            dtype=np.float32
        )

        # ----------------------------------------------------
        # PREDICT ONLY VALID PIXELS
        # ----------------------------------------------------

        if np.any(valid):

            # Flatten valid pixels
            elev_v = elevation[valid]
            slope_v = slope[valid]
            aspect_v = aspect[valid]
            curvature_v = curvature[valid]
            tri_v = tri[valid]
            clay_v = clay[valid]
            sand_v = sand[valid]
            lc_v = landcover[valid]

            # Aspect → sine/cosine
            aspect_rad = np.radians(aspect_v)

            aspect_sin = np.sin(aspect_rad)
            aspect_cos = np.cos(aspect_rad)

            # Landcover one-hot encoding
            lc_10 = (lc_v == 10).astype(np.int8)
            lc_30 = (lc_v == 30).astype(np.int8)
            lc_40 = (lc_v == 40).astype(np.int8)
            lc_50 = (lc_v == 50).astype(np.int8)
            lc_80 = (lc_v == 80).astype(np.int8)

            # ------------------------------------------------
            # CREATE DATAFRAME
            # ------------------------------------------------

            X = pd.DataFrame({
                "elevation": elev_v,
                "slope": slope_v,
                "aspect_sin": aspect_sin,
                "aspect_cos": aspect_cos,
                "curvature": curvature_v,
                "tri": tri_v,
                "clay": clay_v,
                "sand": sand_v,
                "lc_10": lc_10,
                "lc_30": lc_30,
                "lc_40": lc_40,
                "lc_50": lc_50,
                "lc_80": lc_80
            })

            # Make absolutely sure order is correct
            X = X[FEATURES]

            # ------------------------------------------------
            # RANDOM FOREST PROBABILITY
            # ------------------------------------------------

            probabilities = model.predict_proba(X)

            class_1_probability = probabilities[
                :,
                class_1_index
            ]

            # Put probabilities back into raster block
            output[valid] = class_1_probability.astype(
                np.float32
            )

        # ----------------------------------------------------
        # WRITE BLOCK
        # ----------------------------------------------------

        dst.write(
            output,
            1,
            window=window
        )

        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        if (
            block_number % 20 == 0
            or block_number == total_blocks
        ):

            percent = (
                block_number /
                total_blocks
            ) * 100

            print(
                f"Progress: {percent:6.2f}% "
                f"({block_number}/{total_blocks})"
            )


# ============================================================
# CLOSE RASTERS
# ============================================================

for src in sources.values():
    src.close()


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 60)
print("STATIC SUSCEPTIBILITY MAP CREATED")
print("=" * 60)

print(f"\nOutput:")
print(OUTPUT_PATH)

print("\nPixel values:")
print("0.0  = 0% susceptibility")
print("1.0  = 100% susceptibility")

print("\nNodata:")
print("-9999")

print("\nDone.")