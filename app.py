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
    
    # Load SARIMAX
    if os.path.exists("sarimax.pkl"):
        try:
            sarimax_m = joblib.load("sarimax.pkl")
        except Exception as e:
            st.sidebar.warning(f"Failed to load SARIMAX: {e}")
            
    # Load LSTM
    if os.path.exists("lstm_model.keras") and load_model is not None:
        try:
            lstm_m = load_model("lstm_model.keras")
        except Exception as e:
            st.sidebar.warning(f"Failed to load LSTM: {e}")
            
    return sarimax_m, lstm_m

sarimax_model, lstm_model = load_ai_models()

# Cloud-writable scratchpad directory for SQLite
DB_PATH = os.path.join(tempfile.gettempdir(), "solar_data_fleet_v6.db")

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
# DATABASE ORCHESTRATION (Updated Schema & API calls)
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
    
    tomorrow_str = str(date.today() + timedelta(days=1))
    cursor.execute(
        "SELECT COUNT(*) FROM real_time_weather WHERE location = ? AND timestamp LIKE ?", 
        (location_name, f"{tomorrow_str}%")
    )
    if cursor.fetchone()[0] == 0:
        # Corrected Open-Meteo Endpoint with temperature_2m and wind_speed_10m
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
                df.to_sql("real_time_weather", conn, if_exists="append", index=False)
                conn.commit()
        except Exception:
            pass
    conn.close()

def fetch_location_data_from_sql(location_name):
    try:
        conn = sqlite3.connect(DB_PATH)
        tomorrow_str = str(date.today() + timedelta(days=1))
        # Selecting the new wind_speed column for downstream processing
        query = f"SELECT timestamp, temperature, humidity
