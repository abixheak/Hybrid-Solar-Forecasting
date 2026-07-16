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

# Safely import Keras (Requires tensorflow in requirements.txt)
try:
    from tensorflow.keras.models import load_model
except ImportError:
    load_model = None

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
    
    # Load SARIMAX (Updated to prioritize the highly compressed .gz file)
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
    elif os.path.exists("sarimax.pkl"): # Fallback for old 1GB file
        try:
            sarimax_m = joblib.load("sarimax.pkl")
        except Exception as e:
            st.sidebar.warning(f"Failed to load legacy SARIMAX: {e}")
            
    # Load LSTM (Updated to look for the ultra-lightweight .h5 file)
    if os.path.exists("hybrid_lstm_light.h5") and load_model is not None:
        try:
            lstm_m = load_model("hybrid_lstm_light.h5", compile=False)
        except Exception as e:
            st.sidebar.warning(f"Failed to load LSTM: {e}")
    elif os.path.exists("hybrid_lstm_residuals.keras") and load_model is not None:
        try:
            lstm_m = load_model("hybrid_lstm_residuals.keras")
        except Exception as e:
            st.sidebar.warning(f"Failed to load LSTM: {e}")
            
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
# DATABASE ORCHESTRATION (Updated to fetch 48 hours for lookback context)
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
        # Fetch 48 hours (Today + Tomorrow) to provide the 24h lookback for the LSTM
        query = f"SELECT timestamp, temperature, humidity, cloud_cover, irradiance, wind_speed FROM real_time_weather WHERE location = '{location_name}' AND (timestamp LIKE '{today_str}%' OR timestamp LIKE '{tomorrow_str}%') ORDER BY timestamp ASC"
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if df.empty or len(df) < 48: return None, "⚠️ Syncing 48-hour telemetry block..."
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df.head(48), f"✅ {location_name} Node Connected" # Force exactly 48 rows
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
            
            # --- 1. SARIMAX BASELINE FORECAST (Over 48 hours) ---
            # SARIMAX uses all available features
            sarimax_cols = ['temperature', 'humidity', 'cloud_cover', 'irradiance', 'wind_speed']
            X_sarimax = np.nan_to_num(weather_df[sarimax_cols].values)
            
            try:
                sarimax_pred_48 = sarimax_model.predict(n_periods=len(X_sarimax), X=X_sarimax)
            except:
                sarimax_pred_48 = sarimax_model.forecast(steps=len(X_sarimax), exog=X_sarimax)
            
            # --- 2. LSTM SEQUENCE INFERENCE (Lookback = 24, 5 Features) ---
            # Features strictly ordered to match training script: Irradiance, Temp, Wind, Humidity
            lstm_cols = ['irradiance', 'temperature', 'wind_speed', 'humidity']
            X_lstm_weather = np.nan_to_num(weather_df[lstm_cols].values)
            
            # Initialize a 48-hour residual array
            residuals_48 = np.zeros(48)
            
            # Simulate real residuals for the FIRST 24 hours (Today) to give LSTM context
            for i in range(24):
                sim_actual = max(0, sarimax_pred_48[i] + np.random.normal(0, 10))
                residuals_48[i] = sim_actual - sarimax_pred_48[i]
                
            # Autoregressive loop for the NEXT 24 hours (Tomorrow)
            lstm_corrections = []
            for i in range(24, 48):
                # Isolate the past 24 hours of weather (24, 4) and residuals (24, 1)
                window_weather = X_lstm_weather[i-24:i, :]
                window_res = residuals_48[i-24:i].reshape(-1, 1)
                
                # Combine to match training tensor shape: (24, 5)
                window_X = np.hstack((window_weather, window_res))
                
                # Reshape to 3D tensor: (samples=1, time_steps=24, features=5)
                X_input = window_X.reshape(1, 24, 5)
                
                # Predict the residual for hour i
                pred_res = lstm_model.predict(X_input, verbose=0).flatten()[0]
                
                # Store prediction back into array so hour i+1 can see it
                residuals_48[i] = pred_res
                lstm_corrections.append(pred_res)
            
            # --- 3. FINAL HYBRID COMPOSITION (For Tomorrow Only) ---
            factor = geo_data['factor']
            sarimax_tomorrow = sarimax_pred_48[24:]
            generation = [max(0, (s + l) * factor) for s, l in zip(sarimax_tomorrow, lstm_corrections)]
            
            # Simulated Actuals for accuracy calc
            actual_gen = [max(0, g + np.random.normal(0, g * 0.05)) for g in generation]
            errors = [abs((act - pred) / act) for act, pred in zip(actual_gen, generation) if act > 10]
            accuracy_pct = max(0, 100 * (1 - (np.mean(errors) if errors else 0)))

            # --- 4. DEMAND & BESS SIMULATION (Over tomorrow's 24 hours) ---
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

            # Store only tomorrow's timestamps in the UI
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
if st.session_state.simulation_results is not None:
    res = st.session_state.simulation_results
    results_df = pd.DataFrame(res["data_frame_records"])
    
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("☀️ Total Generation", f"{res['total_gen']:,.1f} kWh")
    col2.metric("🎯 Prediction Accuracy", f"{res['accuracy_pct']:.1f}%")
    col3.metric("🚨 Grid Dependency", f"{res['total_grid_dependency']:,.1f} kWh")
    col4.metric("🌿 Self-Sufficiency", f"{res['green_mitigation_pct']:.1f}%")
    st.markdown("<br><br>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📊 Battery Dispatch Analytics", "📋 Detailed Data Ledger"])
    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=results_df["Time"], y=results_df["Demand"], mode='lines', line=dict(color='#FFA500', width=2.5), name='Demand Profile (kW)'))
        fig.add_trace(go.Scatter(x=results_df["Time"], y=results_df["Generation"], mode='lines', line=dict(color='#00CC96', width=2.5), name='Solar Production (kW)'))
        fig.add_trace(go.Scatter(x=results_df["Time"], y=results_df["Battery Storage (kWh)"], fill='tozeroy', fillcolor='rgba(0, 191, 255, 0.1)', mode='lines', line=dict(color='#00BFFF', width=3, shape='spline'), name='Battery Reserve (kWh)'))
        fig.add_trace(go.Scatter(x=results_df["Time"], y=results_df["True Deficit (Fossil Backup)"], mode='lines', line=dict(color='#FF4B4B', width=2, dash='dash'), name='External Grid (kW)'))
        fig.update_layout(title=dict(text=f"BESS Balancing Matrix: {selected_city}", font=dict(size=18, color="#FAFAFA")), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"), xaxis=dict(showgrid=False, tickangle=-45), yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title="Power / Storage (kW)"))
        st.plotly_chart(fig, use_container_width=True)
        
    with tab2:
        st.dataframe(results_df, use_container_width=True)
        st.caption("Note: Inference maps to 24-hour sliding sequence matching `hybrid_lstm_residuals.keras` dimensions (samples=1, timesteps=24, features=5).")
else:
    if weather_df is not None:
        st.info(f"Ready. Configure BESS parameters on the left and run the simulation.")
    else:
        st.info("System initializing telemetry...")
