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

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="SolarNet | Storage Engine",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. CUSTOM CSS INJECTION (The Secret to the UI Upgrade)
st.markdown("""
    <style>
    /* Gradient Title */
    .gradient-text {
        font-size: 2.8rem !important;
        font-weight: 800 !important;
        background: -webkit-linear-gradient(45deg, #00CC96, #00BFFF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px !important;
        padding-bottom: 0px !important;
    }
    
    /* Subtitle styling */
    .sub-text {
        font-size: 1.1rem;
        color: #A0AEC0;
        margin-top: -10px;
        margin-bottom: 30px;
    }

    /* Floating KPI Cards */
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

    /* Hide Streamlit Branding for a standalone app feel */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Enhance sidebar background */
    [data-testid="stSidebar"] {
        background-color: #0E1117;
        border-right: 1px solid #2D3748;
    }
    </style>
""", unsafe_allow_html=True)

# Cloud-writable scratchpad directory for SQLite
DB_PATH = os.path.join(tempfile.gettempdir(), "solar_data_fleet_v5.db")

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
        
        if df.empty: return None, "⚠️ Syncing telemetry..."
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df, f"✅ {location_name} Node Connected"
    except Exception as e:
        return None, f"Database Error: {str(e)}"

# -----------------------------------------------------------------------------
# APPLICATION INTERFACE LAYOUT
# -----------------------------------------------------------------------------
st.markdown('<p class="gradient-text">SolarNet Microgrid OS</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Automated solar dispatch engine and BESS balancing dashboard.</p>', unsafe_allow_html=True)

# SIDEBAR: Cleaner, grouped UI
st.sidebar.markdown("### 🎛️ Command Center")
selected_city = st.sidebar.selectbox("🎯 Target Grid Node", list(LOCATIONS.keys()))
geo_data = LOCATIONS[selected_city]

if st.session_state.current_active_city != selected_city:
    st.session_state.simulation_results = None
    st.session_state.current_active_city = selected_city

# Grouped Settings inside Expanders
with st.sidebar.expander("🔋 BESS Configuration", expanded=True):
    battery_capacity = st.slider("Storage Capacity (kWh)", 200, 1000, 500, 50)
    initial_charge_pct = st.slider("Initial Charge (SoC %)", 0, 100, 20, 5)

with st.sidebar.expander("📈 Demand Modifications", expanded=False):
    load_scaler = st.slider("Peak Load Modifier", 0.7, 1.5, 1.0, 0.05)
    st.caption("Simulate heatwaves or high-demand events.")

# Initialize and verify database operations
initialize_and_populate_db(selected_city, geo_data['lat'], geo_data['lon'])
weather_df, db_status = fetch_location_data_from_sql(selected_city)
st.sidebar.markdown("---")
st.sidebar.success(db_status)

# -----------------------------------------------------------------------------
# SIMULATION TRIGGER
# -----------------------------------------------------------------------------
if weather_df is not None:
    if st.button(f"⚡ Run Grid Dispatch Simulation for {selected_city}", type="primary", use_container_width=True):
        with st.spinner("Processing neural generation and thermodynamic matrices..."):
            
            # Logic Engine (Unchanged)
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

            current_charge = battery_capacity * (initial_charge_pct / 100.0)
            battery_soc_history = []
            unmet_deficit_history = []
            
            for gen, dem in zip(generation, demand):
                raw_delta = gen - dem
                if raw_delta > 0:
                    available_room = battery_capacity - current_charge
                    energy_to_store = min(raw_delta, available_room)
                    current_charge += energy_to_store
                    unmet_deficit = 0
                else:
                    needed_energy = abs(raw_delta)
                    energy_dispatched = min(needed_energy, current_charge)
                    current_charge -= energy_dispatched
                    unmet_deficit = needed_energy - energy_dispatched
                    
                battery_soc_history.append(current_charge)
                unmet_deficit_history.append(unmet_deficit)

            total_gen = np.trapezoid(generation, dx=1.0)
            total_dem = np.trapezoid(demand, dx=1.0)
            total_grid_dependency = sum(unmet_deficit_history)
            green_mitigation_pct = 100 * (1.0 - (total_grid_dependency / total_dem)) if total_dem > 0 else 100

            st.session_state.simulation_results = {
                "total_gen": total_gen,
                "total_dem": total_dem,
                "total_grid_dependency": total_grid_dependency,
                "green_mitigation_pct": green_mitigation_pct,
                "data_frame_records": {
                    "Time": weather_df["timestamp"].dt.strftime('%H:%M'),
                    "Generation": generation,
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
    
    # 1. Custom Styled KPI Banner
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("☀️ Total Generation", f"{res['total_gen']:,.1f} kWh")
    col2.metric("🔌 Demand Profile", f"{res['total_dem']:,.1f} kWh")
    col3.metric("🚨 Grid Dependency", f"{res['total_grid_dependency']:,.1f} kWh")
    col4.metric("🌿 Self-Sufficiency", f"{res['green_mitigation_pct']:.1f}%")
    
    st.markdown("<br><br>", unsafe_allow_html=True)

    # 2. View Layouts
    tab1, tab2 = st.tabs(["📊 Battery Dispatch Analytics", "📋 Detailed Data Ledger"])
    
    with tab1:
        fig = go.Figure()
        
        # Consumer Demand Profile Line
        fig.add_trace(go.Scatter(
            x=results_df["Time"], y=results_df["Demand"],
            mode='lines', line=dict(color='#FFA500', width=2.5),
            name='Demand Profile (kW)'
        ))
        
        # Solar Array Production Line
        fig.add_trace(go.Scatter(
            x=results_df["Time"], y=results_df["Generation"],
            mode='lines', line=dict(color='#00CC96', width=2.5),
            name='Solar Production (kW)'
        ))

        # Battery Storage Area
        fig.add_trace(go.Scatter(
            x=results_df["Time"], y=results_df["Battery Storage (kWh)"],
            fill='tozeroy', fillcolor='rgba(0, 191, 255, 0.1)',
            mode='lines', line=dict(color='#00BFFF', width=3, shape='spline'),
            name='Battery Reserve (kWh)'
        ))
        
        # Unmet Deficit
        fig.add_trace(go.Scatter(
            x=results_df["Time"], y=results_df["True Deficit (Fossil Backup)"],
            mode='lines', line=dict(color='#FF4B4B', width=2, dash='dash'),
            name='External Grid (kW)'
        ))
        
        fig.update_layout(
            title=dict(text=f"BESS Balancing Matrix: {selected_city}", font=dict(size=18, color="#FAFAFA")),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(showgrid=False, tickangle=-45),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title="Power / Storage (kW)")
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with tab2:
        st.dataframe(results_df, use_container_width=True)
else:
    if weather_df is not None:
        st.info(f"Ready. Configure BESS parameters on the left and run the simulation.")
    else:
        st.info("System initializing telemetry...")
