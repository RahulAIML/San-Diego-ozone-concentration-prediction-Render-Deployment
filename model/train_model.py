import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
import joblib
import os

# --- Configuration ---
DATA_PATH = 'd:/Ozone_Project_7th_dec/final_cal.csv'
MODEL_DIR = 'model_artifacts'
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Feature Mapping (Shortname -> Actual Column Name) ---
# Based on cols.txt
FEATURE_MAP = {
    # Target
    'target': 'ozone_ppb',
    
    # Tier 1
    'ozone_lag_1': 'ozone_lag_1',
    'tmax': 'tmax',
    'tavg': 'tavg',
    'CUTI': 'CUTI Coastal Upwelling Transport Index (m^2/s)',
    'month_sin': 'month_sin',
    'month_cos': 'month_cos',
    'wspd': 'wspd',
    'Tmax_inland': 'Tmax_inland',
    'land_sea_temp_diff': 'land_sea_temp_diff',
    
    # Tier 2
    'CUTI_lag1': 'CUTI Coastal Upwelling Transport Index (m^2/s)_lag1',
    'CUTI_lag3': 'CUTI Coastal Upwelling Transport Index (m^2/s)_lag3',
    'CUTI_roll7_mean': 'CUTI Coastal Upwelling Transport Index (m^2/s)_roll7_mean',
    'thermal_stability': 'thermal_stability',
    'marine_layer_presence': 'marine_layer_presence',
    'BEUTI': 'BEUTI Biological Effective Upwelling Transport Index (mmol m/s)',
    'tsun': 'tsun',
    'temp_range': 'temp_range',
    
    # Tier 3
    'CUTI_lag7': 'CUTI Coastal Upwelling Transport Index (m^2/s)_lag7',
    'sst_value_sst': 'sst_value_sst',
    'sst_anomaly': 'sst_anomaly',
    'distance_to_coast_km': 'distance_to_coast_km',
    
    # For Regime Analysis (Clustering)
    'regime_features': [
        'CUTI Coastal Upwelling Transport Index (m^2/s)',
        'land_sea_temp_diff',
        'tmax',
        'wspd'
    ]
}

def load_and_preprocess_data():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    
    # Select Features
    features = [v for k, v in FEATURE_MAP.items() if k != 'target' and k != 'regime_features']
    target = FEATURE_MAP['target']
    
    # Ensure all columns exist
    missing_cols = [c for c in features + [target] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
    
    # Create Interaction Term (Tier 3)
    # tmax * CUTI
    cuti_col = FEATURE_MAP['CUTI']
    df['tmax_x_CUTI'] = df['tmax'] * df[cuti_col]
    features.append('tmax_x_CUTI')
    
    # Filter Data
    # Drop rows where target is missing
    df = df.dropna(subset=[target])
    
    X = df[features]
    y = df[target]
    
    # Train/Test Split (Time-based: 2023 is test, rest is train)
    # Assuming 'year' column exists and is reliable.
    if 'year' in df.columns:
        train_mask = df['year'] < 2023
        test_mask = df['year'] == 2023
        
        if test_mask.sum() == 0:
            print("Warning: No data for 2023 found. Using random split.")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        else:
            X_train = X[train_mask]
            y_train = y[train_mask]
            X_test = X[test_mask]
            y_test = y[test_mask]
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
    return X_train, X_test, y_train, y_test, features

def train():
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data()
    
    # --- Imputation ---
    print("Imputing missing values...")
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    
    # --- Regime Analysis (K-Means) ---
    print("Performing Regime Analysis (K-Means)...")
    # We use a subset of features for clustering to identify regimes
    # Need to map the 'regime_features' names to indices in X_train_imputed
    # Or just re-select them from X_train (but we need imputed values)
    
    # Let's create a separate scaler/kmeans pipeline for just the regime features
    # to make it easy to use in inference.
    regime_cols = FEATURE_MAP['regime_features']
    # Get indices of regime cols in feature_names
    regime_indices = [feature_names.index(col) for col in regime_cols]
    
    X_train_regime = X_train_imputed[:, regime_indices]
    
    scaler_regime = StandardScaler()
    X_train_regime_scaled = scaler_regime.fit_transform(X_train_regime)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(X_train_regime_scaled)
    
    # Save Regime Artifacts
    joblib.dump(kmeans, os.path.join(MODEL_DIR, 'kmeans.pkl'))
    joblib.dump(scaler_regime, os.path.join(MODEL_DIR, 'scaler_regime.pkl'))
    joblib.dump(regime_cols, os.path.join(MODEL_DIR, 'regime_cols.pkl'))
    
    # Add Cluster ID as a feature? 
    # The proposal says "Characterize clusters", but doesn't explicitly say to use them as features for the main model.
    # However, it might help. Let's add it as a one-hot encoded feature or just leave it for analysis.
    # For now, we will just use the Tier 3 features as planned.
    
    # --- Main Model Training ---
    print("Training XGBoost Model...")
    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        objective='reg:squarederror',
        n_jobs=-1,
        random_state=42
    )
    
    model.fit(X_train_imputed, y_train)
    
    # --- Evaluation ---
    print("Evaluating...")
    y_pred = model.predict(X_test_imputed)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    
    # --- Save Artifacts ---
    print("Saving artifacts...")
    joblib.dump(model, os.path.join(MODEL_DIR, 'ozone_model.pkl'))
    joblib.dump(imputer, os.path.join(MODEL_DIR, 'imputer.pkl'))
    joblib.dump(feature_names, os.path.join(MODEL_DIR, 'feature_names.pkl'))
    
    # Save Metrics
    import json
    metrics = {
        "rmse": float(rmse),
        "r2_score": float(r2)
    }
    
    # Save to model_artifacts (for backend)
    with open(os.path.join(MODEL_DIR, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
        
    # Save to dedicated metrics folder (for visibility)
    metrics_dir = 'd:/Ozone_Project_7th_dec/metrics'
    os.makedirs(metrics_dir, exist_ok=True)
    with open(os.path.join(metrics_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print(f"Metrics saved to {os.path.join(MODEL_DIR, 'metrics.json')} and {os.path.join(metrics_dir, 'metrics.json')}")
    
    print("Done.")

if __name__ == "__main__":    # starting point of any code file
    train()
