# app.py
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import requests
import os
import tempfile
import joblib
from datetime import date, timedelta
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAXResults

# Safely import Keras and apply the robust from_config override
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.layers import Dense
    
    # Override BOTH initialization and deserialization configs to strip the bug
    class SafeDense(Dense):
        def __init__(self, *args, **kwargs):
            kwargs.pop('quantization_config', None)
            super().__init__(*args, **kwargs)
            
        @classmethod
        def from_config(cls, config):
            # This intercepts the config dict BEFORE Keras tries to build the layer
            config.pop('quantization_config', None)
            return cls(**config)

except ImportError:
    load_model = None
    SafeDense = None

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="SolarNet | Storage Engine",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. CUSTOM CSS INJECTION
st.markdown("""
    <style>
    .gradient-text {
        font-size: 2.8rem !important;
        font-weight: 800 !important;
        background: -webkit-linear-gradient(45deg, #00CC96, #00BFFF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px !important;
        padding-bottom: 0px !important;
    }
    .sub-text {
        font-size: 1.1rem;
        color: #A0AEC0;
        margin-top: -10px;
        margin-bottom: 30px;
    }
    div[data-testid="stMetric"] {
        background-color: #171923;
        border: 1px solid #2D3748;
        padding: 15px 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.4);
        transition: transform 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        border-color: #00CC96;
    }
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stSidebar"] {
        background-color: #0E1117;
        border-right: 1px solid #2D3748;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# MACHINE LEARNING MODEL LOADER (Runs Once at Startup)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_ai_models():
    """Loads offline-trained SARIMAX and LSTM models into global memory cache."""
    sarimax_m = None
    lstm_m = None
    
    # Load SARIMAX
    if os.path.exists("sarimax_compressed.pkl.gz"):
        try:
            sarimax_m = joblib.load("sarimax_compressed.pkl.gz")
        except Exception as e:
            st.sidebar.warning(f"Failed to load compressed SARIMAX: {e}")
    elif os.path.exists("sarimax_light.pkl"):
        try:
            sarimax_m = SARIMAXResults.load("sarimax_light.pkl")
        except Exception as e:
            st.sidebar.warning(f"Failed to load lightweight SARIMAX: {e}")
    elif os.path.exists("sarimax.pkl"):
        try:
            sarimax_m = joblib.load("sarimax.pkl")
        except Exception as e:
            st.sidebar.warning(f"Failed to load legacy SARIMAX: {e}")
            
    # Inject the robust SafeDense into Keras's deserializer
    custom_objs = {'Dense': SafeDense} if SafeDense else {}
            
    # Load LSTM (Native Keras deserialization is now protected by custom_objects)
    if os.path.exists("hybrid_lstm_light.h5") and load_model is not None:
        try:
            lstm_m = load_model("hybrid_lstm_light.h5", compile=False, custom_objects=custom_objs)
        except Exception as e:
            st.sidebar.warning(f"Failed to load LSTM .h5: {e}")
    elif os.path.exists("hybrid_lstm_residuals.keras") and load_model is not None:
        try:
            lstm_m = load_model("hybrid_lstm_residuals.keras", compile=False, custom_objects=custom_objs)
        except Exception as e:
            st.sidebar.warning(f"Failed to load LSTM .keras: {e}")
            
    return sarimax_m, lstm_m

sarimax_model, lstm_model = load_ai_models()

# Cloud-writable scratchpad directory for SQLite
DB_PATH = os.path.join(tempfile.gettempdir(), "solar_data_fleet_v7.db")

# ENTERPRISE REGIONAL REGISTRY
LOCATIONS = {
    "Chennai": {"lat": 13.0827, "lon": 80.2707, "factor": 1.0, "base_demand": 220},
    "New Delhi": {"lat": 28.6139, "lon": 77.2090, "factor": 1.15, "base_demand": 280},
    "Mumbai": {"lat": 19.0760, "lon": 72.8777, "factor": 0.90, "base_demand": 260},
    "Bengaluru": {"lat": 12.9716, "lon": 77.5946, "factor": 1.05, "base_demand": 190}
}

# -----------------------------------------------------------------------------
# PERSISTENT SESSION STATE INITIALIZATION
# -----------------------------------------------------------------------------
if "simulation_results" not in st.session_state:
    st.session_state.simulation_results = None
if "current_active_city" not in st.session_state:
    st.session_state.current_active_city = None

# -----------------------------------------------------------------------------
# DATABASE ORCHESTRATION
# -----------------------------------------------------------------------------
def initialize_and_populate_db(location_name, lat, lon):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS real_time_weather (
            location TEXT, timestamp DATETIME, temperature REAL,
            humidity REAL, cloud_cover REAL, irradiance REAL, wind_speed REAL,
            PRIMARY KEY (location, timestamp)
        )
    ''')
    conn.commit()
    
    today_str = str(date.today())
    tomorrow_str = str(date.today() + timedelta(days=1))
    
    cursor.execute(
        "SELECT COUNT(*) FROM real_time_weather WHERE location = ? AND (timestamp LIKE ? OR timestamp LIKE ?)", 
        (location_name, f"{today_str}%", f"{tomorrow_str}%")
    )
    if cursor.fetchone()[0] < 48:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m,relative_humidity_2m,cloud_cover,direct_radiation,wind_speed_10m&past_days=1&forecast_days=2"
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame({
                    "location": location_name,
                    "timestamp": pd.to_datetime(data["hourly"]["time"]),
                    "temperature": data["hourly"]["temperature_2m"],
                    "humidity": data["hourly"]["relative_humidity_2m"],
                    "cloud_cover": data["hourly"]["cloud_cover"],
                    "irradiance": data["hourly"]["direct_radiation"],
                    "wind_speed": data["hourly"]["wind_speed_10m"]
                })
                df.to_sql("real_time_weather", conn, if_exists="replace", index=False)
                conn.commit()
        except Exception:
            pass
    conn.close()

def fetch_location_data_from_sql(location_name):
    try:
        conn = sqlite3.connect(DB_PATH)
        today_str = str(date.today())
        tomorrow_str = str(date.today() + timedelta(days=1))
        query = f"SELECT timestamp, temperature, humidity, cloud_cover, irradiance, wind_speed FROM real_time_weather WHERE location = '{location_name}' AND (timestamp LIKE '{today_str}%' OR timestamp LIKE '{tomorrow_str}%') ORDER BY timestamp ASC"
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if df.empty or len(df) < 48: return None, "⚠️ Syncing 48-hour telemetry block..."
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df.head(48), f"✅ {location_name} Node Connected" 
    except Exception as e:
        return None, f"Database Error: {str(e)}"

# -----------------------------------------------------------------------------
# APPLICATION INTERFACE LAYOUT
# -----------------------------------------------------------------------------
st.markdown('<p class="gradient-text">SolarNet Microgrid OS</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Automated solar dispatch engine and BESS balancing dashboard.</p>', unsafe_allow_html=True)

st.sidebar.markdown("### 🎛️ Command Center")
selected_city = st.sidebar.selectbox("🎯 Target Grid Node", list(LOCATIONS.keys()))
geo_data = LOCATIONS[selected_city]

if st.session_state.current_active_city != selected_city:
    st.session_state.simulation_results = None
    st.session_state.current_active_city = selected_city

with st.sidebar.expander("🔋 BESS Configuration", expanded=True):
    battery_capacity = st.slider("Storage Capacity (kWh)", 200, 1000, 500, 50)
    initial_charge_pct = st.slider("Initial Charge (SoC %)", 0, 100, 20, 5)

with st.sidebar.expander("📈 Demand Modifications", expanded=False):
    load_scaler = st.slider("Peak Load Modifier", 0.7, 1.5, 1.0, 0.05)
    st.caption("Simulate heatwaves or high-demand events.")

if sarimax_model is None or lstm_model is None:
    st.sidebar.error("⚠️ AI Models offline. Check root directory.")
else:
    st.sidebar.success("🤖 Hybrid AI Models Online")

initialize_and_populate_db(selected_city, geo_data['lat'], geo_data['lon'])
weather_df, db_status = fetch_location_data_from_sql(selected_city)
st.sidebar.markdown("---")
st.sidebar.success(db_status)

# -----------------------------------------------------------------------------
# SIMULATION TRIGGER & AI INFERENCE ENGINE
# -----------------------------------------------------------------------------
if weather_df is not None:
    if st.button(f"⚡ Run AI Grid Dispatch Simulation for {selected_city}", type="primary", use_container_width=True):
        
        if sarimax_model is None or lstm_model is None:
            st.error("❌ Cannot run simulation. Pre-trained models ('sarimax_compressed.pkl.gz', 'hybrid_lstm_light.h5') were not found.")
            st.stop()
            
        with st.spinner("Executing sequence-based autoregressive inference..."):
            
            sarimax_cols = ['temperature', 'humidity', 'cloud_cover', 'irradiance', 'wind_speed']
            X_sarimax = np.nan_to_num(weather_df[sarimax_cols].values)
            
            try:
                sarimax_pred_48 = sarimax_model.predict(n_periods=len(X_sarimax), X=X_sarimax)
            except:
                sarimax_pred_48 = sarimax_model.forecast(steps=len(X_sarimax), exog=X_sarimax)
            
            lstm_cols = ['irradiance', 'temperature', 'wind_speed', 'humidity']
            X_lstm_weather = np.nan_to_num(weather_df[lstm_cols].values)
            
            residuals_48 = np.zeros(48)
            
            for i in range(24):
                sim_actual = max(0, sarimax_pred_48[i] + np.random.normal(0, 10))
                residuals_48[i] = sim_actual - sarimax_pred_48[i]
                
            lstm_corrections = []
            for i in range(24, 48):
                window_weather = X_lstm_weather[i-24:i, :]
                window_res = residuals_48[i-24:i].reshape(-1, 1)
                
                window_X = np.hstack((window_weather, window_res))
                X_input = window_X.reshape(1, 24, 5)
                
                pred_res = lstm_model.predict(X_input, verbose=0).flatten()[0]
                residuals_48[i] = pred_res
                lstm_corrections.append(pred_res)
            
            factor = geo_data['factor']
            sarimax_tomorrow = sarimax_pred_48[24:]
            generation = [max(0, (s + l) * factor) for s, l in zip(sarimax_tomorrow, lstm_corrections)]
            
            actual_gen = [max(0, g + np.random.normal(0, g * 0.05)) for g in generation]
            errors = [abs((act - pred) / act) for act, pred in zip(actual_gen, generation) if act > 10]
            accuracy_pct = max(0, 100 * (1 - (np.mean(errors) if errors else 0)))

            base_load = geo_data['base_demand'] * load_scaler
            demand = []
            for hour in range(24):
                morning_peak = 0.4 * np.exp(-((hour - 9) / 2.5) ** 2)
                evening_peak = 0.7 * np.exp(-((hour - 19) / 3.0) ** 2)
                night_dip = 0.2 if (hour < 5 or hour > 22) else 0.35
                demand.append(max(20, base_load * (night_dip + morning_peak + evening_peak + np.random.uniform(-0.02, 0.02))))

            current_charge = battery_capacity * (initial_charge_pct / 100.0)
            battery_soc_history = []
            unmet_deficit_history = []
            
            for gen, dem in zip(generation, demand):
                raw_delta = gen - dem
                if raw_delta > 0:
                    current_charge += min(raw_delta, battery_capacity - current_charge)
                    unmet_deficit = 0
                else:
                    needed = abs(raw_delta)
                    dispatched = min(needed, current_charge)
                    current_charge -= dispatched
                    unmet_deficit = needed - dispatched
                    
                battery_soc_history.append(current_charge)
                unmet_deficit_history.append(unmet_deficit)

            total_gen = np.trapezoid(generation, dx=1.0)
            total_dem = np.trapezoid(demand, dx=1.0)
            total_grid_dependency = sum(unmet_deficit_history)
            green_mitigation_pct = 100 * (1.0 - (total_grid_dependency / total_dem)) if total_dem > 0 else 100

            tomorrow_df = weather_df.iloc[24:].reset_index(drop=True)
            st.session_state.simulation_results = {
                "total_gen": total_gen,
                "total_dem": total_dem,
                "total_grid_dependency": total_grid_dependency,
                "green_mitigation_pct": green_mitigation_pct,
                "accuracy_pct": accuracy_pct,
                "data_frame_records": {
                    "Time": tomorrow_df["timestamp"].dt.strftime('%H:%M'),
                    "Generation": generation,
                    "Actual (Simulated)": actual_gen,
                    "Demand": demand,
                    "Battery Storage (kWh)": battery_soc_history,
                    "True Deficit (Fossil Backup)": unmet_deficit_history
                }
            }

# -----------------------------------------------------------------------------
# RENDERING LAYER
# -----------------------------------------------------------------------------
if st.session_state.simulation_
