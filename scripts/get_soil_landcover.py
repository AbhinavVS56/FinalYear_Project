import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling, transform_bounds
import geopandas as gpd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DEM_PATH = os.path.join(
    PROJECT_ROOT,
    "datasets", "processed", "dem",
    "idukki_dem_utm43n.tif"
)

BOUNDARY_PATH = os.path.join(
    PROJECT_ROOT,
    "datasets", "raw", "boundaries",
    "idukki_boundary.gpkg"
)

TRAINING_CSV = os.path.join(
    PROJECT_ROOT,
    "datasets", "processed", "training",
    "idukki_training_dataset.csv"
)

OUT_SOIL_DIR = os.path.join(
    PROJECT_ROOT,
    "datasets", "processed", "soil"
)

OUT_LANDCOVER_DIR = os.path.join(
    PROJECT_ROOT,
    "datasets", "processed", "landcover"
)

OUT_TRAINING_DIR = os.path.join(
    PROJECT_ROOT,
    "datasets", "processed", "training"
)

RAW_LANDCOVER_DIR = os.path.join(
    PROJECT_ROOT,
    "datasets", "raw", "landcover"
)


for folder in [
    OUT_SOIL_DIR,
    OUT_LANDCOVER_DIR,
    OUT_TRAINING_DIR,
    RAW_LANDCOVER_DIR
]:
    os.makedirs(folder, exist_ok=True)


# ============================================================
# ESA WORLDCOVER CLASS CODES
# ============================================================

WORLDCOVER_CLASSES = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / sparse vegetation",
    70: "Snow and ice",
    80: "Permanent water bodies",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen"
}


# ============================================================
# LOAD DEM GRID
# ============================================================

def load_dem_grid():

    with rasterio.open(DEM_PATH) as dem:

        return {
            "transform": dem.transform,
            "crs": dem.crs,
            "width": dem.width,
            "height": dem.height,
            "bounds": dem.bounds,
            "profile": dem.profile
        }


# ============================================================
# LOAD IDUKKI BOUNDARY
# ============================================================

def load_boundary():

    gdf = gpd.read_file(BOUNDARY_PATH)

    gdf_4326 = gdf.to_crs("EPSG:4326")

    return gdf, gdf_4326


# ============================================================
# SOILGRIDS
# ============================================================

def download_soilgrids_variable(var_name, grid):

    """
    Download SoilGrids 0-5 cm mean clay/sand and
    reproject directly onto the DEM grid.
    """

    url = (
        "/vsicurl/https://files.isric.org/soilgrids/latest/data/"
        f"{var_name}/{var_name}_0-5cm_mean.vrt"
    )

    print(
        f"Fetching SoilGrids '{var_name}' "
        f"(0-5cm mean)..."
    )

    dst_array = np.full(
        (grid["height"], grid["width"]),
        np.nan,
        dtype="float32"
    )

    with rasterio.Env(
        GDAL_HTTP_MAX_RETRY="5",
        GDAL_HTTP_RETRY_DELAY="2"
    ):

        with rasterio.open(url) as src:

            reproject(
                source=rasterio.band(src, 1),
                destination=dst_array,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=grid["transform"],
                dst_crs=grid["crs"],
                resampling=Resampling.bilinear,
                src_nodata=src.nodata,
                dst_nodata=np.nan
            )

    # SoilGrids stores clay/sand as g/kg * 10
    # Convert to percentage
    valid = ~np.isnan(dst_array)

    dst_array[valid] = dst_array[valid] / 10.0

    out_path = os.path.join(
        OUT_SOIL_DIR,
        f"idukki_{var_name}_0_5cm.tif"
    )

    profile = grid["profile"].copy()

    profile.update(
        dtype="float32",
        count=1,
        nodata=np.nan
    )

    with rasterio.open(
        out_path,
        "w",
        **profile
    ) as dst:

        dst.write(dst_array, 1)

    print(
        f"  -> saved: {out_path}"
    )

    print(
        f"  -> range: "
        f"{np.nanmin(dst_array):.1f}% - "
        f"{np.nanmax(dst_array):.1f}% "
        f"(mean {np.nanmean(dst_array):.1f}%)"
    )

    return out_path


# ============================================================
# WORLDCOVER TILE
# ============================================================

def get_worldcover_tile():

    """
    Idukki is completely covered by the N09E075
    ESA WorldCover 2021 tile.
    """

    tile_name = "N09E075"

    out_path = os.path.join(
        RAW_LANDCOVER_DIR,
        "ESA_WorldCover_10m_2021_v200_N09E075_Map.tif"
    )

    if os.path.exists(out_path):

        print(
            "WorldCover tile already exists."
        )

        return out_path

    print(
        "WorldCover tile not found."
    )

    print(
        "Downloading N09E075..."
    )

    try:

        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config

        key = (
            "v200/2021/map/"
            "ESA_WorldCover_10m_2021_v200_"
            "N09E075_Map.tif"
        )

        s3 = boto3.client(
            "s3",
            config=Config(
                signature_version=UNSIGNED
            ),
            region_name="eu-central-1"
        )

        s3.download_file(
            "esa-worldcover",
            key,
            out_path
        )

    except Exception as e:

        print(
            f"boto3 failed: {e}"
        )

        print(
            "Trying HTTPS download..."
        )

        import urllib.request

        url = (
            "https://esa-worldcover.s3.eu-central-1.amazonaws.com/"
            "v200/2021/map/"
            "ESA_WorldCover_10m_2021_v200_N09E075_Map.tif"
        )

        urllib.request.urlretrieve(
            url,
            out_path
        )

    print(
        f"  -> saved: {out_path}"
    )

    return out_path


# ============================================================
# WORLDCOVER - MEMORY SAFE PROCESSING
# ============================================================

def process_worldcover(grid, boundary_4326):

    """
    Processes only the part of the WorldCover tile
    covering Idukki.

    IMPORTANT:
    We DO NOT mosaic/load the entire 36,000 x 36,000
    WorldCover tile into RAM.

    Instead, we reproject directly from the source
    tile into the target DEM grid.
    """

    worldcover_path = get_worldcover_tile()

    print(
        "\nProcessing WorldCover using memory-safe "
        "windowed reprojection..."
    )

    # --------------------------------------------------------
    # Determine DEM bounding box in WorldCover CRS
    # --------------------------------------------------------

    with rasterio.open(worldcover_path) as src:

        source_bounds = transform_bounds(
            grid["crs"],
            src.crs,
            grid["bounds"].left,
            grid["bounds"].bottom,
            grid["bounds"].right,
            grid["bounds"].top
        )

        print(
            "WorldCover source CRS:",
            src.crs
        )

        print(
            "Required source bounds:",
            source_bounds
        )

        # ----------------------------------------------------
        # Create output array matching DEM
        # ----------------------------------------------------

        destination = np.zeros(
            (grid["height"], grid["width"]),
            dtype=np.uint8
        )

        # ----------------------------------------------------
        # Direct reprojection
        # ----------------------------------------------------

        reproject(
            source=rasterio.band(src, 1),
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=grid["transform"],
            dst_crs=grid["crs"],
            resampling=Resampling.nearest,
            src_nodata=src.nodata,
            dst_nodata=0
        )

    # --------------------------------------------------------
    # Save aligned WorldCover
    # --------------------------------------------------------

    out_path = os.path.join(
        OUT_LANDCOVER_DIR,
        "idukki_worldcover2021.tif"
    )

    profile = grid["profile"].copy()

    profile.update(
        dtype="uint8",
        count=1,
        nodata=0
    )

    with rasterio.open(
        out_path,
        "w",
        **profile
    ) as dst:

        dst.write(destination, 1)

    print(
        f"  -> saved: {out_path}"
    )

    # --------------------------------------------------------
    # Class distribution
    # --------------------------------------------------------

    valid = destination != 0

    values, counts = np.unique(
        destination[valid],
        return_counts=True
    )

    total = counts.sum()

    print(
        "\nWorldCover class distribution "
        "over Idukki:"
    )

    for value, count in sorted(
        zip(values, counts),
        key=lambda x: -x[1]
    ):

        name = WORLDCOVER_CLASSES.get(
            int(value),
            "Unknown"
        )

        percentage = (
            100 * count / total
        )

        print(
            f"  {int(value):>4} "
            f"{name:<28} "
            f"{count:>10} px "
            f"({percentage:5.2f}%)"
        )

    return out_path


# ============================================================
# SAMPLE RASTER AT TRAINING POINTS
# ============================================================

def sample_raster_at_points(
    raster_path,
    xs,
    ys
):

    with rasterio.open(raster_path) as src:

        values = list(
            src.sample(
                zip(xs, ys)
            )
        )

    return np.array(
        [v[0] for v in values]
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("STEP 0: Loading DEM grid + boundary")
    print("=" * 60)

    grid = load_dem_grid()

    boundary_native, boundary_4326 = (
        load_boundary()
    )

    bounds_4326 = (
        boundary_4326.total_bounds
    )

    print(
        f"DEM grid: "
        f"{grid['width']} x "
        f"{grid['height']}"
    )

    print(
        f"DEM CRS: {grid['crs']}"
    )

    print(
        f"Idukki bounds: {bounds_4326}"
    )


    # ========================================================
    # STEP 1 - SOIL
    # ========================================================

    print("\n" + "=" * 60)
    print("STEP 1: SoilGrids clay + sand")
    print("=" * 60)

    clay_path = download_soilgrids_variable(
        "clay",
        grid
    )

    sand_path = download_soilgrids_variable(
        "sand",
        grid
    )


    # ========================================================
    # STEP 2 - LAND COVER
    # ========================================================

    print("\n" + "=" * 60)
    print("STEP 2: ESA WorldCover 2021")
    print("=" * 60)

    worldcover_path = process_worldcover(
        grid,
        boundary_4326
    )


    # ========================================================
    # STEP 3 - TRAINING DATA
    # ========================================================

    print("\n" + "=" * 60)
    print(
        "STEP 3: Sampling new layers "
        "at existing training points"
    )
    print("=" * 60)

    df = pd.read_csv(
        TRAINING_CSV
    )

    print(
        f"Loaded {len(df)} training points."
    )

    # --------------------------------------------------------
    # Sample soil
    # --------------------------------------------------------

    print("Sampling clay...")

    df["clay"] = sample_raster_at_points(
        clay_path,
        df["x"],
        df["y"]
    )

    print("Sampling sand...")

    df["sand"] = sample_raster_at_points(
        sand_path,
        df["x"],
        df["y"]
    )

    # --------------------------------------------------------
    # Sample land cover
    # --------------------------------------------------------

    print("Sampling land cover...")

    df["landcover"] = sample_raster_at_points(
        worldcover_path,
        df["x"],
        df["y"]
    )

    # --------------------------------------------------------
    # Remove invalid samples
    # --------------------------------------------------------

    before = len(df)

    df = df.dropna(
        subset=[
            "clay",
            "sand"
        ]
    )

    df = df[
        df["landcover"] != 0
    ]

    after = len(df)

    if before != after:

        print(
            f"Dropped "
            f"{before - after} "
            f"invalid samples."
        )


    # ========================================================
    # SAVE V2 DATASET
    # ========================================================

    output_csv = os.path.join(
        OUT_TRAINING_DIR,
        "idukki_training_dataset_v2.csv"
    )

    df.to_csv(
        output_csv,
        index=False
    )

    print(
        "\nUpgraded dataset saved:"
    )

    print(
        output_csv
    )

    print(
        f"\nFinal rows: {len(df)}"
    )

    print(
        "\nColumns:"
    )

    print(
        list(df.columns)
    )


    # ========================================================
    # LAND COVER DISTRIBUTION
    # ========================================================

    print(
        "\nLand cover distribution "
        "at training points:"
    )

    distribution = (
        df["landcover"]
        .value_counts()
        .sort_index()
    )

    for value, count in distribution.items():

        name = WORLDCOVER_CLASSES.get(
            int(value),
            "Unknown"
        )

        percentage = (
            100 * count / len(df)
        )

        print(
            f"  {int(value):>4} "
            f"{name:<28} "
            f"{count:>6} "
            f"({percentage:5.2f}%)"
        )


    # ========================================================
    # DONE
    # ========================================================

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)

    print(
        "\nNext step:"
    )

    print(
        "One-hot encode landcover and "
        "train the upgraded Random Forest "
        "using the SAME spatial block split."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()