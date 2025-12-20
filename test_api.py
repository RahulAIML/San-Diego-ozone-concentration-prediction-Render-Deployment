import requests
import json

url = "http://localhost:8000/predict"
data = {
    "ozone_lag_1": 40,
    "tmax": 25,
    "tavg": 20,
    "CUTI": 0.5,
    "month_sin": 0.5,
    "month_cos": 0.5,
    "wspd": 10,
    "Tmax_inland": 30,
    "land_sea_temp_diff": 5,
    "CUTI_lag1": 0.5,
    "CUTI_lag3": 0.5,
    "CUTI_roll7_mean": 0.5,
    "thermal_stability": 0.1,
    "marine_layer_presence": 0,
    "BEUTI": 10,
    "tsun": 10,
    "temp_range": 10,
    "CUTI_lag7": 0.5,
    "sst_value_sst": 18,
    "sst_anomaly": 0,
    "distance_to_coast_km": 5
}

try:
    response = requests.post(url, json=data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
except Exception as e:
    print(f"Error: {e}")
