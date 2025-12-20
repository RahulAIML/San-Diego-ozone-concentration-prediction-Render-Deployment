from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import PredictionLog
import json
import pandas as pd
import numpy as np
import joblib
import os
from django.conf import settings

# --- Load Artifacts ---
MODEL_DIR = 'd:/Ozone_Project_7th_dec/model_artifacts'
model_artifacts = {}

try:
    model_artifacts['model'] = joblib.load(os.path.join(MODEL_DIR, 'ozone_model.pkl'))
    model_artifacts['imputer'] = joblib.load(os.path.join(MODEL_DIR, 'imputer.pkl'))
    model_artifacts['feature_names'] = joblib.load(os.path.join(MODEL_DIR, 'feature_names.pkl'))
    model_artifacts['kmeans'] = joblib.load(os.path.join(MODEL_DIR, 'kmeans.pkl'))
    model_artifacts['scaler_regime'] = joblib.load(os.path.join(MODEL_DIR, 'scaler_regime.pkl'))
    model_artifacts['regime_cols'] = joblib.load(os.path.join(MODEL_DIR, 'regime_cols.pkl'))
    print("Model artifacts loaded successfully.")
except Exception as e:
    print(f"Error loading model artifacts: {e}")

def get_aqi_category(ozone_ppb):
    if ozone_ppb <= 54: return "Good"
    elif ozone_ppb <= 70: return "Moderate"
    elif ozone_ppb <= 85: return "Unhealthy for Sensitive Groups"
    elif ozone_ppb <= 105: return "Unhealthy"
    elif ozone_ppb <= 200: return "Very Unhealthy"
    else: return "Hazardous"

def get_regime_description(cluster_id):
    mapping = {
        0: "Regime A (Potential Marine Influence)",
        1: "Regime B (Potential Stagnation)",
        2: "Regime C (Transitional)"
    }
    return mapping.get(cluster_id, "Unknown")


@api_view(['GET'])
def health_check(request):
    """
    Simple health check endpoint to verify API is running.
    """
    return Response({"status": "healthy", "message": "Django API is running!"})

@api_view(['POST'])
def predict(request):
    """
    Endpoint that accepts input data, mocks a prediction, 
    stores both in the database, and returns the result.
    """
    input_data = request.data
    
    if 'model' in model_artifacts:
        try:
            # Real Prediction Logic
            feature_names = model_artifacts['feature_names']
            imputer = model_artifacts['imputer']
            model = model_artifacts['model']
            
            # Create DataFrame
            df = pd.DataFrame([input_data])
            
            # Interaction Term
            if 'tmax' in df.columns and 'CUTI' in df.columns:
                df['tmax_x_CUTI'] = pd.to_numeric(df['tmax']) * pd.to_numeric(df['CUTI'])
            
            # Prepare Input
            model_input = pd.DataFrame(index=[0])
            for col in feature_names:
                if col in df.columns:
                    model_input[col] = pd.to_numeric(df[col], errors='coerce')
                else:
                    model_input[col] = np.nan
            
            # Impute
            input_imputed = imputer.transform(model_input)
            
            # Predict Ozone
            prediction_val = model.predict(input_imputed)[0]
            
            # Regime Logic
            regime_cols = model_artifacts['regime_cols']
            regime_indices = [feature_names.index(col) for col in regime_cols]
            regime_input = input_imputed[:, regime_indices]
            regime_input_scaled = model_artifacts['scaler_regime'].transform(regime_input)
            cluster = model_artifacts['kmeans'].predict(regime_input_scaled)[0]
            
            predicted_output = {
                "predicted_ozone": float(prediction_val),
                "air_quality_category": get_aqi_category(prediction_val),
                "regime_cluster": int(cluster),
                "regime_description": get_regime_description(cluster),
                "source": "ML Model"
            }
        except Exception as e:
            print(f"Prediction error: {e}")
            predicted_output = {
                "error": str(e),
                "source": "Error Fallback"
            }
    else:
        # Mock Fallback
        predicted_output = {
            "prediction_score": 0.95,
            "class": "Ozone Normal",
            "mock_note": "Model not loaded. Using mock.",
            "source": "Mock"
        }
    
    # Store in SQLite database
    # Converting dicts to string for TextField storage
    log = PredictionLog.objects.create(
        input_data=json.dumps(input_data),
        predicted_output=json.dumps(predicted_output)
    )
    
    return Response({
        "status": "success",
        "input_recieved": input_data,
        "prediction": predicted_output,
        "log_id": log.id
    })

@api_view(['GET'])
def get_logs(request):
    """
    Retrieve recent prediction logs.
    """
    logs = PredictionLog.objects.all().order_by('-created_at')[:50]
    data = []
    for log in logs:
        try:
            in_d = json.loads(log.input_data)
        except:
            in_d = log.input_data
            
        try:
            out_d = json.loads(log.predicted_output)
        except:
            out_d = log.predicted_output
            
        data.append({
            "id": log.id,
            "created_at": log.created_at,
            "input": in_d,
            "output": out_d
        })
    return Response(data)
