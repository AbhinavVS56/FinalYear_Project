import rasterio
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path("../datasets/processed/terrain_features")

files = {
    "tri": BASE / "idukki_tri.tif",
    "curvature": BASE / "idukki_curvature.tif",
    "slope": BASE / "idukki_slope.tif"
}

def inspect_feature(name, path, n=15):
    print("\n" + "=" * 70)
    print(f"{name.upper()} - TOP {n} EXTREME VALUES")
    print("=" * 70)

    with rasterio.open(path) as src:
        data = src.read(1).astype(float)
        transform = src.transform
        nodata = src.nodata

        valid = np.isfinite(data)

        if nodata is not None:
            valid &= data != nodata

        # Ignore zero values
        valid &= data != 0

        rows, cols = np.where(valid)
        values = data[rows, cols]

        if name == "curvature":
            indices = np.argsort(np.abs(values))[::-1][:n]
        else:
            indices = np.argsort(values)[::-1][:n]

        records = []

        for i in indices:
            row = rows[i]
            col = cols[i]
            value = values[i]

            x, y = rasterio.transform.xy(
                transform,
                row,
                col,
                offset="center"
            )

            records.append({
                "rank": len(records) + 1,
                "row": row,
                "col": col,
                "x_utm": round(x, 2),
                "y_utm": round(y, 2),
                "value": value
            })

        df = pd.DataFrame(records)

        print(df.to_string(index=False))

        print("\nStatistics:")
        print(f"Minimum : {np.nanmin(data):.6f}")
        print(f"Maximum : {np.nanmax(data):.6f}")
        print(f"Mean    : {np.nanmean(data):.6f}")
        print(f"Std     : {np.nanstd(data):.6f}")


inspect_feature("tri", files["tri"], 20)
inspect_feature("curvature", files["curvature"], 20)
inspect_feature("slope", files["slope"], 20)

print("\n" + "=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)