import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import os

# --- Configuration ---
DATA_PATH = 'd:/Ozone_Project_7th_dec/final_cal.csv'
MODEL_DIR = 'model_artifacts' # Relative to this script or absolute? train_model uses relative 'model_artifacts'
# Since we will run this from d:/Ozone_Project_7th_dec/model/ or d:/Ozone_Project_7th_dec/, we need to be careful.
# Let's assume we run from d:/Ozone_Project_7th_dec/ like: python model/generate_plots.py
# But train_model.py had MODEL_DIR = 'model_artifacts', implying it expected to be run from model/ or it created it in CWD.
# Let's use absolute paths to be safe.
BASE_DIR = 'd:/Ozone_Project_7th_dec'
MODEL_DIR = os.path.join(BASE_DIR, 'model', 'model_artifacts')
REPORT_PLOTS_DIR = os.path.join(BASE_DIR, 'report_plots')
os.makedirs(REPORT_PLOTS_DIR, exist_ok=True)

# --- Feature Mapping (Same as train_model.py) ---
FEATURE_MAP = {
    'target': 'ozone_ppb',
    'ozone_lag_1': 'ozone_lag_1',
    'tmax': 'tmax',
    'tavg': 'tavg',
    'CUTI': 'CUTI Coastal Upwelling Transport Index (m^2/s)',
    'month_sin': 'month_sin',
    'month_cos': 'month_cos',
    'wspd': 'wspd',
    'Tmax_inland': 'Tmax_inland',
    'land_sea_temp_diff': 'land_sea_temp_diff',
    'CUTI_lag1': 'CUTI Coastal Upwelling Transport Index (m^2/s)_lag1',
    'CUTI_lag3': 'CUTI Coastal Upwelling Transport Index (m^2/s)_lag3',
    'CUTI_roll7_mean': 'CUTI Coastal Upwelling Transport Index (m^2/s)_roll7_mean',
    'thermal_stability': 'thermal_stability',
    'marine_layer_presence': 'marine_layer_presence',
    'BEUTI': 'BEUTI Biological Effective Upwelling Transport Index (mmol m/s)',
    'tsun': 'tsun',
    'temp_range': 'temp_range',
    'CUTI_lag7': 'CUTI Coastal Upwelling Transport Index (m^2/s)_lag7',
    'sst_value_sst': 'sst_value_sst',
    'sst_anomaly': 'sst_anomaly',
    'distance_to_coast_km': 'distance_to_coast_km',
    'regime_features': [
        'CUTI Coastal Upwelling Transport Index (m^2/s)',
        'land_sea_temp_diff',
        'tmax',
        'wspd'
    ]
}

def load_data():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    
    # Select Features
    features = [v for k, v in FEATURE_MAP.items() if k != 'target' and k != 'regime_features']
    target = FEATURE_MAP['target']
    
    # Create Interaction Term (Tier 3)
    cuti_col = FEATURE_MAP['CUTI']
    if cuti_col in df.columns and 'tmax' in df.columns:
        df['tmax_x_CUTI'] = df['tmax'] * df[cuti_col]
        features.append('tmax_x_CUTI')
    
    # Filter Data
    df = df.dropna(subset=[target])
    
    return df, features, target

def plot_eda(df, target_col):
    print("Generating EDA plots...")
    
    # 1. Target Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df[target_col], kde=True, bins=30)
    plt.title('Distribution of Ozone Concentration (ppb)')
    plt.xlabel('Ozone (ppb)')
    plt.savefig(os.path.join(REPORT_PLOTS_DIR, 'eda_target_distribution.png'))
    plt.close()
    
    # 2. Correlation Heatmap (Top 15 features)
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    # Get top correlations with target
    if target_col in corr.columns:
        cols = corr.nlargest(15, target_col)[target_col].index
        cm = np.corrcoef(df[cols].values.T)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, 
                    yticklabels=cols.values, xticklabels=cols.values)
        plt.title('Correlation Matrix (Top 15 Correlated Features)')
        plt.tight_layout()
        plt.savefig(os.path.join(REPORT_PLOTS_DIR, 'eda_correlation_heatmap.png'))
        plt.close()

    # 3. Time Series (if Year/Month available)
    if 'year' in df.columns and 'month' in df.columns:
        # Create a rough date for plotting
        df['Date_Approx'] = pd.to_datetime(df[['year', 'month']].assign(DAY=1))
        # Group by date to handle duplicates if any
        daily_avg = df.groupby('Date_Approx')[target_col].mean().reset_index()
        plt.figure(figsize=(14, 6))
        plt.plot(daily_avg['Date_Approx'], daily_avg[target_col])
        plt.title('Ozone Concentration Over Time (Monthly Average)')
        plt.xlabel('Date')
        plt.ylabel('Ozone (ppb)')
        plt.savefig(os.path.join(REPORT_PLOTS_DIR, 'eda_time_series.png'))
        plt.close()

def plot_feature_engineering(df, features):
    print("Generating Feature Engineering plots...")
    
    # Load Regime Analysis Artifacts
    try:
        kmeans = joblib.load(os.path.join(MODEL_DIR, 'kmeans.pkl'))
        scaler_regime = joblib.load(os.path.join(MODEL_DIR, 'scaler_regime.pkl'))
        regime_cols = joblib.load(os.path.join(MODEL_DIR, 'regime_cols.pkl'))
        
        # Prepare data for clustering visualization
        # We need to impute if there are NaNs, as KMeans doesn't handle them
        imputer = SimpleImputer(strategy='mean')
        X_regime = df[regime_cols]
        X_regime_imputed = imputer.fit_transform(X_regime)
        X_regime_scaled = scaler_regime.transform(X_regime_imputed)
        
        clusters = kmeans.predict(X_regime_scaled)
        df['Cluster'] = clusters
        
        # Plot Clusters (using first two regime features)
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=df[regime_cols[0]], y=df[regime_cols[1]], hue=df['Cluster'], palette='viridis', alpha=0.6)
        plt.title(f'Regime Clusters: {regime_cols[0]} vs {regime_cols[1]}')
        plt.xlabel(regime_cols[0])
        plt.ylabel(regime_cols[1])
        plt.savefig(os.path.join(REPORT_PLOTS_DIR, 'fe_clusters.png'))
        plt.close()
        
    except Exception as e:
        print(f"Skipping Regime Analysis plots: {e}")

def plot_model_performance(df, features, target_col):
    print("Generating Model Performance plots...")
    
    # Load Model Artifacts
    try:
        model = joblib.load(os.path.join(MODEL_DIR, 'ozone_model.pkl'))
        imputer = joblib.load(os.path.join(MODEL_DIR, 'imputer.pkl'))
        # feature_names = joblib.load(os.path.join(MODEL_DIR, 'feature_names.pkl')) # Optional
        
        # Prepare Test Data (2023)
        if 'year' in df.columns:
            test_mask = df['year'] == 2023
            if test_mask.sum() > 0:
                X_test = df.loc[test_mask, features]
                y_test = df.loc[test_mask, target_col]
            else:
                # Fallback to random split
                _, X_test, _, y_test = train_test_split(df[features], df[target_col], test_size=0.2, random_state=42)
        else:
             _, X_test, _, y_test = train_test_split(df[features], df[target_col], test_size=0.2, random_state=42)
        
        X_test_imputed = imputer.transform(X_test)
        y_pred = model.predict(X_test_imputed)
        
        # 1. Actual vs Predicted
        plt.figure(figsize=(8, 8))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Actual vs Predicted Ozone Levels')
        plt.savefig(os.path.join(REPORT_PLOTS_DIR, 'model_actual_vs_predicted.png'))
        plt.close()
        
        # 2. Residuals
        residuals = y_test - y_pred
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plt.savefig(os.path.join(REPORT_PLOTS_DIR, 'model_residuals.png'))
        plt.close()
        
        # 3. Residual Histogram
        plt.figure(figsize=(10, 6))
        sns.histplot(residuals, kde=True)
        plt.title('Distribution of Residuals')
        plt.xlabel('Residual')
        plt.savefig(os.path.join(REPORT_PLOTS_DIR, 'model_residual_hist.png'))
        plt.close()
        
        # 4. Feature Importance
        plt.figure(figsize=(12, 8))
        xgb.plot_importance(model, max_num_features=15, height=0.5)
        plt.title('Feature Importance (Top 15)')
        plt.tight_layout()
        plt.savefig(os.path.join(REPORT_PLOTS_DIR, 'model_feature_importance.png'))
        plt.close()
        
        return X_test, y_test, imputer # Return for overfitting check if needed
        
    except Exception as e:
        print(f"Error generating model plots: {e}")
        return None, None, None

def plot_overfitting_check(df, features, target_col):
    print("Generating Overfitting Check (Learning Curve)...")
    
    # We need to instantiate a new model with the same params to run learning_curve
    # Or we can just use the loaded model class if we want.
    # Let's use a fresh estimator for the learning curve
    
    X = df[features]
    y = df[target_col]
    
    # Impute whole dataset for learning curve
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    model = xgb.XGBRegressor(
        n_estimators=100, # Reduced for speed in learning curve
        learning_rate=0.05,
        max_depth=6,
        objective='reg:squarederror',
        n_jobs=-1,
        random_state=42
    )
    
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_imputed, y, cv=3, scoring='neg_root_mean_squared_error',
        n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5)
    )
    
    train_scores_mean = -np.mean(train_scores, axis=1)
    test_scores_mean = -np.mean(test_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.xlabel("Training examples")
    plt.ylabel("RMSE")
    plt.title("Learning Curve")
    plt.legend(loc="best")
    plt.grid()
    plt.savefig(os.path.join(REPORT_PLOTS_DIR, 'overfitting_learning_curve.png'))
    plt.close()

def main():
    df, features, target = load_data()
    
    plot_eda(df, target)
    plot_feature_engineering(df, features)
    plot_model_performance(df, features, target)
    plot_overfitting_check(df, features, target)
    
    print(f"All plots saved to {REPORT_PLOTS_DIR}")

if __name__ == "__main__":
    main()
