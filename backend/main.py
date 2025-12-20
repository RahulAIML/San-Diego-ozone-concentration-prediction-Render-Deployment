from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Ozone Ocean Prediction API")

# Enable CORS for Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load Artifacts ---
MODEL_DIR = 'd:/Ozone_Project_7th_dec/model_artifacts'
try:
    model = joblib.load(os.path.join(MODEL_DIR, 'ozone_model.pkl'))
    imputer = joblib.load(os.path.join(MODEL_DIR, 'imputer.pkl'))
    feature_names = joblib.load(os.path.join(MODEL_DIR, 'feature_names.pkl'))
    kmeans = joblib.load(os.path.join(MODEL_DIR, 'kmeans.pkl'))
    scaler_regime = joblib.load(os.path.join(MODEL_DIR, 'scaler_regime.pkl'))
    regime_cols = joblib.load(os.path.join(MODEL_DIR, 'regime_cols.pkl'))
except Exception as e:
    print(f"Error loading model artifacts: {e}")
    # For development, we might want to continue even if models fail to load, 
    # but for production this should be fatal.
    pass

class PredictionRequest(BaseModel):
    # Tier 1
    ozone_lag_1: float
    tmax: float
    tavg: float
    CUTI: float
    month_sin: float
    month_cos: float
    wspd: float
    Tmax_inland: float
    land_sea_temp_diff: float
    
    # Tier 2
    CUTI_lag1: float
    CUTI_lag3: float
    CUTI_roll7_mean: float
    thermal_stability: float
    marine_layer_presence: int
    BEUTI: float
    tsun: float
    temp_range: float
    
    # Tier 3
    CUTI_lag7: float
    sst_value_sst: float
    sst_anomaly: float
    distance_to_coast_km: float

class PredictionResponse(BaseModel):
    predicted_ozone: float
    air_quality_category: str
    regime_cluster: int
    regime_description: str

def get_aqi_category(ozone_ppb):
    if ozone_ppb <= 54:
        return "Good"
    elif ozone_ppb <= 70:
        return "Moderate"
    elif ozone_ppb <= 85:
        return "Unhealthy for Sensitive Groups"
    elif ozone_ppb <= 105:
        return "Unhealthy"
    elif ozone_ppb <= 200:
        return "Very Unhealthy"
    else:
        return "Hazardous"

def get_regime_description(cluster_id):
    # These labels are hypothetical based on the proposal's "Regime Analysis" section.
    # Ideally, we would inspect the cluster centers to assign these labels accurately.
    # For now, we assign generic labels.
    mapping = {
        0: "Regime A (Potential Marine Influence)",
        1: "Regime B (Potential Stagnation)",
        2: "Regime C (Transitional)"
    }
    return mapping.get(cluster_id, "Unknown")

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        data = request.dict()
        
        # Create DataFrame
        df = pd.DataFrame([data])
        
        # Feature Engineering (Interaction Term)
        df['tmax_x_CUTI'] = df['tmax'] * df['CUTI']
        
        # Ensure correct column order for model
        # We need to match 'feature_names' exactly
        # Some features might be missing if we didn't include them in the Request model
        # but the Imputer expects them?
        # Actually, the Imputer expects the exact columns it was trained on.
        # The 'feature_names' list we saved contains the columns passed to the imputer.
        
        # Let's check if we have all features.
        # The Request model has most of them.
        # If any are missing (e.g. if we forgot some in the Pydantic model), we should add them as NaN.
        
        # Reorder/Fill
        model_input = pd.DataFrame(index=[0])
        for col in feature_names:
            if col in df.columns:
                model_input[col] = df[col]
            else:
                model_input[col] = np.nan
        
        # Impute
        input_imputed = imputer.transform(model_input)
        
        # Predict Ozone
        prediction = model.predict(input_imputed)[0]
        
        # Predict Regime
        # We need to extract the specific columns for regime analysis
        # regime_cols contains the names.
        # We need to find their indices in the 'feature_names' list to get them from 'input_imputed'
        # OR we can just extract them from 'model_input' (but we need imputed values)
        
        regime_indices = [feature_names.index(col) for col in regime_cols]
        regime_input = input_imputed[:, regime_indices]
        
        regime_input_scaled = scaler_regime.transform(regime_input)
        cluster = kmeans.predict(regime_input_scaled)[0]
        
        return PredictionResponse(
            predicted_ozone=float(prediction),
            air_quality_category=get_aqi_category(prediction),
            regime_cluster=int(cluster),
            regime_description=get_regime_description(cluster)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Ozone Ocean Prediction API is running"}
