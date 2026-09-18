import requests
import pandas as pd
from pathlib import Path

# Idukki, Kerala - representative coordinate
LATITUDE = 9.9189
LONGITUDE = 77.1025

url = "https://api.open-meteo.com/v1/forecast"

params = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "hourly": "rain,precipitation",
    "forecast_days": 1,
    "timezone": "Asia/Kolkata"
}

response = requests.get(url, params=params)
response.raise_for_status()

data = response.json()

df = pd.DataFrame({
    "time": data["hourly"]["time"],
    "rain_mm": data["hourly"]["rain"],
    "precipitation_mm": data["hourly"]["precipitation"]
})

# Save output
output_dir = Path("../datasets/processed/rainfall")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "idukki_live_rainfall.csv"
df.to_csv(output_file, index=False)

print("\nLIVE IDUKKI RAINFALL")
print("=" * 40)
print(df.to_string(index=False))

print("\nSaved to:")
print(output_file)