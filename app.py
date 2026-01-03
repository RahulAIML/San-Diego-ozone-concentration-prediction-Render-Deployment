import os
import json
import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import database  # Import the new DB module

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model_artifacts')

app = Flask(__name__, static_folder=os.path.join(BASE_DIR, 'frontend/dist'))
CORS(app)

# --- Database Setup ---
database.init_db()

# --- Load Artifacts ---
model_artifacts = {}
try:
    model_artifacts['model'] = joblib.load(os.path.join(MODEL_DIR, 'ozone_model.pkl'))
    model_artifacts['imputer'] = joblib.load(os.path.join(MODEL_DIR, 'imputer.pkl'))
    model_artifacts['feature_names'] = joblib.load(os.path.join(MODEL_DIR, 'feature_names.pkl'))
    model_artifacts['kmeans'] = joblib.load(os.path.join(MODEL_DIR, 'kmeans.pkl'))
    model_artifacts['scaler_regime'] = joblib.load(os.path.join(MODEL_DIR, 'scaler_regime.pkl'))
    model_artifacts['regime_cols'] = joblib.load(os.path.join(MODEL_DIR, 'regime_cols.pkl'))
    app.logger.info("Model artifacts loaded successfully.")
except Exception as e:
    app.logger.error(f"Error loading model artifacts: {e}")

# --- Helper Functions ---
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

# --- Routes ---

@app.route('/api/health')
def health_check():
    return jsonify({"status": "healthy", "message": "Flask API is running!"})

@app.route('/api/predict/', methods=['POST'])
def predict():
    input_data = request.json
    predicted_output = {}

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
            app.logger.error(f"Prediction error: {e}")
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
    
    # Store in Database
    log_id = database.insert_log(input_data, predicted_output)

    return jsonify({
        "status": "success",
        "input_recieved": input_data,
        "prediction": predicted_output,
        "log_id": log_id
    })

@app.route('/api/logs/', methods=['GET'])
def get_logs():
    data = database.fetch_logs(limit=50)
    if data is not None:
        return jsonify(data)
    else:
        return jsonify({"error": "Failed to fetch logs"}), 500

# --- Serve Frontend ---
@app.route('/')
def serve_index():
    if os.path.exists(os.path.join(app.static_folder, 'index.html')):
        return send_from_directory(app.static_folder, 'index.html')
    return "Frontend not built.", 404

@app.route('/<path:path>')
def serve_static(path):
    if os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    return serve_index()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
