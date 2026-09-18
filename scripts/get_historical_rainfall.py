import requests
import pandas as pd
from pathlib import Path

# Representative location in Idukki
LATITUDE = 9.9189
LONGITUDE = 77.1025

url = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "start_date": "2018-06-01",
    "end_date": "2018-08-31",
    "daily": "precipitation_sum,rain_sum",
    "timezone": "Asia/Kolkata"
}

response = requests.get(url, params=params)
response.raise_for_status()

data = response.json()

df = pd.DataFrame({
    "date": data["daily"]["time"],
    "precipitation_mm": data["daily"]["precipitation_sum"],
    "rain_mm": data["daily"]["rain_sum"]
})

output_dir = Path("../datasets/processed/rainfall")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "idukki_historical_rainfall_2018.csv"

df.to_csv(output_file, index=False)

print("\nHISTORICAL IDUKKI RAINFALL")
print("=" * 50)
print(df.to_string(index=False))

print("\nTotal days:", len(df))
print("Total rainfall:", round(df["rain_mm"].sum(), 2), "mm")
print("Maximum daily rainfall:", round(df["rain_mm"].max(), 2), "mm")

print("\nSaved to:")
print(output_file)