import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
import datetime

# Try importing ML libraries safely (in case local/cloud setups differ)
try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

try:
    from tensorflow.keras.models import load_model
    HAS_TF = True
except ImportError:
    HAS_TF = False

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Advanced Hybrid Solar Forecasting",
    page_icon="☀️",
    layout="wide"
)

# Custom metric styling
st.markdown("""
<style>
div[data-testid="stMetric"] {
    background-color: #1E1E1E;
    border: 1px solid #333;
    border-radius: 8px;
    padding: 15px;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.5);
}
</style>
""", unsafe_index=True)

# -----------------------------------------------------------------------------
# 2. MODEL LOADING (WITH RELATIVE PATHS & FALLBACKS)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_forecasting_models():
    # Relative paths suitable for deployment or local execution folders
    sarimax_path = os.path.join("models", "saved_models", "sarimax_model.pkl")
    lstm_path = os.path.join("models", "saved_models", "lstm_model.keras")
    
    models = {"sarimax": None, "lstm": None, "status": "Loaded Successfully"}
    
    # Check for SARIMAX
    if os.path.exists(sarimax_path) and HAS_JOBLIB:
        try:
            models["sarimax"] = joblib.load(sarimax_path)
        except Exception as e:
            models["status"] = f"Error loading SARIMAX: {str(e)}"
    else:
        models["status"] = "Missing Model Files (Using Simulation Mode)"
        
    # Check for LSTM
    if os.path.exists(lstm_path) and HAS_TF:
        try:
            models["lstm"] = load_model(lstm_path)
        except Exception as e:
            models["status"] = f"Error loading LSTM: {str(e)}"
            
    return models

loaded_models = load_forecasting_models()

# -----------------------------------------------------------------------------
# 3. LIVE WEATHER DATA FETCHING (OPEN-METEO API)
# -----------------------------------------------------------------------------
def fetch_live_weather(lat, lon):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_20m,relative_humidity_2m,cloud_cover,direct_radiation&forecast_days=1"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame({
                "Timestamp": pd.to_datetime(data["hourly"]["time"]),
                "Temperature (°C)": data["hourly"]["temperature_20m"],
                "Humidity (%)": data["hourly"]["relative_humidity_2m"],
                "Cloud Cover (%)": data["hourly"]["cloud_cover"],
                "Irradiance (W/m²)": data["hourly"]["direct_radiation"]
            })
            return df, "Live API data synchronized."
    except Exception:
        pass
    
    # Fallback simulated data if API fails or rate-limited
    times = [datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + datetime.timedelta(hours=i) for i in range(24)]
    df = pd.DataFrame({
        "Timestamp": times,
        "Temperature (°C)": [24 + 6 * np.sin(i/24 * 2 * np.pi - np.pi/2) for i in range(24)],
        "Humidity (%)": [60 - 20 * np.sin(i/24 * 2 * np.pi - np.pi/2) for i in range(24)],
        "Cloud Cover (%)": [20 + 15 * np.cos(i/12) for i in range(24)],
        "Irradiance (W/m²)": [max(0, 800 * np.sin(i/24 * np.pi)) if 6 <= i <= 18 else 0 for i in range(24)]
    })
    return df, "API unavailable. Using microgrid climate simulator fallback."

# -----------------------------------------------------------------------------
# 4. SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
st.sidebar.title("Forecast Controls")

# Display System Health Status
if loaded_models["status"] == "Loaded Successfully":
    st.sidebar.success(f"🟢 Status: {loaded_models['status']}")
    sim_mode = False
else:
    st.sidebar.warning(f"🟡 Status: {loaded_models['status']}")
    sim_mode = st.sidebar.checkbox("Enable Academic Demonstration Mode", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Future Weather Coordinates")
lat = st.sidebar.number_input("Latitude", value=13.0827, format="%.4f") # Default: Chennai region context
lon = st.sidebar.number_input("Longitude", value=80.2707, format="%.4f")

# Fetch live conditions
weather_df, log_msg = fetch_live_weather(lat, lon)
st.sidebar.info(log_msg)

# -----------------------------------------------------------------------------
# 5. MAIN DASHBOARD CONTENT
# -----------------------------------------------------------------------------
st.title("☀️ Advanced Hybrid Solar Forecasting")
st.subheader("SARIMAX Statistical Baseline + Deep Learning LSTM Residual Correction")

if st.button("🚀 Generate 24-Hour Horizon Forecast", type="primary"):
    with st.spinner("Processing dual-stage inference pipeline..."):
        
        # Core data extraction
        timestamps = weather_df["Timestamp"]
        irradiance = weather_df["Irradiance (W/m²)"]
        
        if not sim_mode and loaded_models["sarimax"] is not None:
            # 1. Linear Forecast Stage via SARIMAX
            sarimax_pred = loaded_models["sarimax"].forecast(steps=24)
            
            # 2. Non-Linear Residual Correction Stage via LSTM
            # Create features matrix matching original training scaler shapes
            features = weather_df[["Temperature (°C)", "Humidity (%)", "Cloud Cover (%)"]].values
            features_scaled = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-5)
            lstm_input = np.expand_dims(features_scaled, axis=0) # Add batch axis
            
            if loaded_models["lstm"] is not None:
                lstm_residuals = loaded_models["lstm"].predict(lstm_input).flatten()
            else:
                lstm_residuals = np
