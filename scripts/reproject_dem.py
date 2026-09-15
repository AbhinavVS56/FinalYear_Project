import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

input_path = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "dem"
    / "idukki_dem_clipped.tif"
)

output_path = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "dem"
    / "idukki_dem_utm43n.tif"
)

target_crs = "EPSG:32643"

with rasterio.open(input_path) as src:

    print("========== SOURCE DEM ==========")
    print("CRS:", src.crs)
    print("Resolution:", src.res)
    print("Size:", src.width, "x", src.height)

    transform, width, height = calculate_default_transform(
        src.crs,
        target_crs,
        src.width,
        src.height,
        *src.bounds
    )

    profile = src.profile.copy()

    profile.update({
        "crs": target_crs,
        "transform": transform,
        "width": width,
        "height": height
    })

    print("\n========== REPROJECTING ==========")
    print("Target CRS:", target_crs)

    with rasterio.open(output_path, "w", **profile) as dst:

        reproject(
            source=rasterio.band(src, 1),
            destination=rasterio.band(dst, 1),
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=target_crs,
            resampling=Resampling.bilinear
        )

print("\n========== COMPLETE ==========")
print("Output:", output_path)
print("New size:", width, "x", height)
print("New resolution:", transform.a, abs(transform.e))
print("CRS:", target_crs)