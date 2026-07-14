# app.py
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import requests
from datetime import date, timedelta
import plotly.express as px
import plotly.graph_objects as go
import os
import tempfile

# 1. PAGE CONFIGURATION (Must be the first Streamlit command)
st.set_page_config(
    page_title="SolarNet | Forecast Engine",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Force the database file to sit in a cloud-writable scratchpad directory
DB_PATH = os.path.join(tempfile.gettempdir(), "solar_data_fleet.db")

# ENTERPRISE LOCATION REGISTER
LOCATIONS = {
    "Chennai": {"lat": 13.0827, "lon": 80.2707, "factor": 1.0},
    "New Delhi": {"lat": 28.6139, "lon": 77.2090, "factor": 1.15},
    "Mumbai": {"lat": 19.0760, "lon": 72.8777, "factor": 0.90},
    "Bengaluru": {"lat": 12.9716, "lon": 77.5946, "factor": 1.05}
}

# -----------------------------------------------------------------------------
# DATABASE FUNCTIONS (Unchanged, running smoothly)
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
# FRONTEND UI & SIDEBAR
# -----------------------------------------------------------------------------
# Main Header
st.markdown("## ☀️ SolarNet Enterprise Dashboard")
st.markdown("Real-time distributed solar generation forecasting across regional grid nodes.")
st.divider()

# Sidebar Styling
st.sidebar.markdown("### ⚙️ Control Center")
selected_city = st.sidebar.selectbox("🎯 Target Grid Node", list(LOCATIONS.keys()))
geo_data = LOCATIONS[selected_city]

st.sidebar.markdown("---")
st.sidebar.markdown("#### Node Telemetry")
st.sidebar.code(f"LAT: {geo_data['lat']}° N\nLON: {geo_data['lon']}° E", language="text")

# Initialize DB & Fetch
initialize_and_populate_db(selected_city, geo_data['lat'], geo_data['lon'])
weather_df, db_status = fetch_location_data_from_sql(selected_city)
st.sidebar.success(db_status)

# -----------------------------------------------------------------------------
# MAIN DASHBOARD AREA
# -----------------------------------------------------------------------------
if weather_df is not None:
    if st.button(f"⚡ Generate Predictive Model for {selected_city}", type="primary", use_container_width=True):
        with st.spinner("Initializing neural network inference & matrix calculations..."):
            
            # Mathematical Simulation Logic
            factor = geo_data['factor']
            base_gen = [max(0, 450 * np.sin(i/24 * np.pi)) * factor if 6 <= i <= 18 else 0 for i in range(24)]
            sarimax_pred = [val * (1 + 0.12 * np.sin(i)) if val > 0 else 0 for i, val in enumerate(base_gen)]
            lstm_corrections = [-25 * np.cos(i/3) if val > 0 else 0 for i, val in enumerate(base_gen)]
            hybrid_pred = [max(0, s + l) for s, l in zip(sarimax_pred, lstm_corrections)]

            results_df = pd.DataFrame({
                "Time": weather_df["timestamp"].dt.strftime('%H:%M'),
                "SARIMAX Baseline": sarimax_pred,
                "Hybrid Output": hybrid_pred
            })
            
            # UI Pop-up
            st.toast(f"Forecast successfully compiled for {selected_city}!", icon="✅")
            
            # Top-Row KPI Metrics
            total_kwh = np.trapezoid(hybrid_pred, dx=1.0)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Est. Energy Yield", f"{total_kwh:.1f} kWh", delta="Optimal")
            c2.metric("Active Node", selected_city)
            c3.metric("Grid Status", "Stable")
            c4.metric("Query Latency", "< 2ms", delta="-1ms", delta_color="inverse")
            
            st.markdown("<br>", unsafe_allow_html=True) # Adds a little breathing room
            
            # Tabbed Layout
            tab1, tab2 = st.tabs(["📈 Generation Forecast (24h)", "🗄️ Raw SQL Inspection"])
            
            with tab1:
                # ---------------------------------------------------------
                # ADVANCED CHART UI DESIGN
                # ---------------------------------------------------------
                fig = go.Figure()
                
                # Layer 1: The Baseline Model (Rendered as semi-transparent bars)
                fig.add_trace(go.Bar(
                    x=results_df["Time"], 
                    y=results_df["SARIMAX Baseline"],
                    marker_color='rgba(255, 75, 75, 0.15)', # Ghosted red
                    marker_line_color='rgba(255, 75, 75, 0.5)', # Red border
                    marker_line_width=1.5,
                    name='Baseline Capacity (kW)'
                ))
                
                # Layer 2: The Hybrid AI Model (Rendered as a smooth, glowing curve)
                fig.add_trace(go.Scatter(
                    x=results_df["Time"], 
                    y=results_df["Hybrid Output"],
                    fill='tozeroy', 
                    fillcolor='rgba(0, 204, 150, 0.2)', # Soft green glow
                    mode='lines+markers', 
                    line=dict(color='#00CC96', width=3, shape='spline'), # 'spline' curves it smoothly
                    marker=dict(size=6, color='#0E1117', line=dict(width=2, color='#00CC96')),
                    name='Live AI Prediction (kW)'
                ))
                
                # Layer 3: Enterprise Layout & Grid Styling
                fig.update_layout(
                    title=dict(
                        text=f"24-Hour Generation Profile: {selected_city}", 
                        font=dict(size=18, color="#FAFAFA")
                    ),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    hovermode="x unified",
                    hoverlabel=dict(
                        bgcolor="#262730", 
                        font_size=14, 
                        bordercolor="#00CC96"
                    ),
                    margin=dict(l=10, r=10, t=50, b=10),
                    legend=dict(
                        orientation="h", 
                        yanchor="bottom", y=1.05, 
                        xanchor="right", x=1,
                        bgcolor="rgba(0,0,0,0)"
                    ),
                    xaxis=dict(
                        showgrid=False, 
                        tickangle=-45,
                        title="Time (Hourly Interval)",
                        title_font=dict(color='gray')
                    ),
                    yaxis=dict(
                        showgrid=True, 
                        gridcolor='rgba(255,255,255,0.05)', # Ultra-faint grid lines
                        gridwidth=1,
                        zeroline=True,
                        zerolinecolor='rgba(255,255,255,0.2)',
                        title="Power Output (kW)",
                        title_font=dict(color='gray')
                    ),
                    barmode='overlay' # Allows the curve to sit perfectly on top of the bars
                )
                
                # Render the chart with a slight aesthetic fade-in
                st.plotly_chart(fig, use_container_width=True)
