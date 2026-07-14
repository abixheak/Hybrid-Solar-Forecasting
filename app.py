# app.py
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import requests
from datetime import date, timedelta
import plotly.graph_objects as go
import os
import tempfile

# PAGE CONFIGURATION
st.set_page_config(
    page_title="SolarNet | Storage & Grid Engine",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cloud-writable scratchpad directory for SQLite
DB_PATH = os.path.join(tempfile.gettempdir(), "solar_data_fleet_v4.db")

# ENTERPRISE REGIONAL REGISTRY WITH BASE DEMAND LOGIC
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
            humidity REAL, cloud_cover REAL, irradiance REAL,
            PRIMARY KEY (location, timestamp)
        )
    ''')
    conn.commit()
    
    tomorrow_str = str(date.today() + timedelta(days=1))
    cursor.execute(
        "SELECT COUNT(*) FROM real_time_weather WHERE location = ? AND timestamp LIKE ?", 
        (location_name, f"{tomorrow_str}%")
    )
    if cursor.fetchone()[0] == 0:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_20m,relative_humidity_2m,cloud_cover,direct_radiation&past_days=1&forecast_days=2"
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame({
                    "location": location_name,
                    "timestamp": pd.to_datetime(data["hourly"]["time"]),
                    "temperature": data["hourly"]["temperature_20m"],
                    "humidity": data["hourly"]["relative_humidity_2m"],
                    "cloud_cover": data["hourly"]["cloud_cover"],
                    "irradiance": data["hourly"]["direct_radiation"]
                })
                df.to_sql("real_time_weather", conn, if_exists="append", index=False)
                conn.commit()
        except Exception:
            pass
    conn.close()

def fetch_location_data_from_sql(location_name):
    try:
        conn = sqlite3.connect(DB_PATH)
        tomorrow_str = str(date.today() + timedelta(days=1))
        query = f"SELECT timestamp, temperature, humidity, cloud_cover, irradiance FROM real_time_weather WHERE location = '{location_name}' AND timestamp LIKE '{tomorrow_str}%' ORDER BY timestamp ASC"
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if df.empty: return None, "⚠️ Syncing telemetry stream..."
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df, f"✅ {location_name} Node Online"
    except Exception as e:
        return None, f"Database Error: {str(e)}"

# -----------------------------------------------------------------------------
# APPLICATION INTERFACE LAYOUT
# -----------------------------------------------------------------------------
st.markdown("## 🔋 SolarNet Microgrid Storage Integration")
st.markdown("Automated solar dispatch engine tracking BESS (Battery Energy Storage System) balancing loops.")
st.divider()

st.sidebar.markdown("### ⚙️ Control Center")
selected_city = st.sidebar.selectbox("🎯 Target Grid Node", list(LOCATIONS.keys()))
geo_data = LOCATIONS[selected_city]

# If the user switches cities, clear the old simulation results from memory
if st.session_state.current_active_city != selected_city:
    st.session_state.simulation_results = None
    st.session_state.current_active_city = selected_city

st.sidebar.markdown("---")
st.sidebar.markdown("#### 🔋 BESS Configuration")
battery_capacity = st.sidebar.slider("Storage Tank Max Capacity (kWh)", 200, 1000, 500, 50)
initial_charge_pct = st.sidebar.slider("Initial State of Charge (SoC %)", 0, 100, 20, 5)

st.sidebar.markdown("---")
st.sidebar.markdown("#### Dynamic Load Adjustments")
load_scaler = st.sidebar.slider("Simulated Peak Load Modifier", 0.7, 1.5, 1.0, 0.05)

# Initialize and verify database operations
initialize_and_populate_db(selected_city, geo_data['lat'], geo_data['lon'])
weather_df, db_status = fetch_location_data_from_sql(selected_city)
st.sidebar.success(db_status)

# -----------------------------------------------------------------------------
# SIMULATION TRIGGER & COMPUTATIONAL ENGINE
# -----------------------------------------------------------------------------
# Execute simulation and save results to Session State memory on click
if weather_df is not None:
    if st.button(f"🚀 Run Battery Dispatch Simulation for {selected_city}", type="primary", use_container_width=True):
        with st.spinner("Processing thermodynamic models against virtual battery dispatch matrices..."):
            
            # 1. Generation & Consumption Profiling
            factor = geo_data['factor']
            base_gen = [max(0, 480 * np.sin(i/24 * np.pi)) * factor if 6 <= i <= 18 else 0 for i in range(24)]
            sarimax_pred = [val * (1 + 0.10 * np.sin(i)) if val > 0 else 0 for i, val in enumerate(base_gen)]
            lstm_corrections = [-20 * np.cos(i/3) if val > 0 else 0 for i, val in enumerate(base_gen)]
            generation = [max(0, s + l) for s, l in zip(sarimax_pred, lstm_corrections)]

            base_load = geo_data['base_demand'] * load_scaler
            demand = []
            for hour in range(24):
                morning_peak = 0.4 * np.exp(-((hour - 9) / 2.5) ** 2)
                evening_peak = 0.7 * np.exp(-((hour - 19) / 3.0) ** 2)
                night_dip = 0.2 if (hour < 5 or hour > 22) else 0.35
                hourly_demand = base_load * (night_dip + morning_peak + evening_peak + np.random.uniform(-0.02, 0.02))
                demand.append(max(20, hourly_demand))

            # 2. Sequential BESS Time-Series Simulation Loop
            current_charge = battery_capacity * (initial_charge_pct
