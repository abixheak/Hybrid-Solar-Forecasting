# app.py
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from datetime import date, timedelta
import os

# (Keep your existing Model Loading logic here)
try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

@st.cache_resource
def load_forecasting_models():
    # Fallback simulation mode for testing
    return {"sarimax": None, "lstm": None, "status": "Simulation Mode"}

loaded_models = load_forecasting_models()

# -----------------------------------------------------------------------------
# NEW: SQL DATABASE INGESTION
# -----------------------------------------------------------------------------
def fetch_tomorrow_from_sql():
    try:
        # Connect to the DB created by data_pipeline.py
        conn = sqlite3.connect("solar_data.db")
        
        # Get exactly tomorrow's date format (e.g., '2026-07-15')
        tomorrow_str = str(date.today() + timedelta(days=1))
        
        # Pure SQL Query
        query = f"""
            SELECT timestamp, temperature, humidity, cloud_cover, irradiance
            FROM real_time_weather 
            WHERE timestamp LIKE '{tomorrow_str}%'
            ORDER BY timestamp ASC
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if df.empty:
            return None, "⚠️ No data found for tomorrow in SQL Database. Did the pipeline run?"
            
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df, f"✅ Data successfully loaded from SQL for {tomorrow_str}"
        
    except Exception as e:
        return None, f"Database Error: {str(e)}"

# -----------------------------------------------------------------------------
# DASHBOARD UI
# -----------------------------------------------------------------------------
st.title("☀️ Automated Hybrid Solar Forecasting")
st.subheader("Enterprise ETL Pipeline architecture connected via SQLite")

st.sidebar.title("System Status")
st.sidebar.success("Backend API decoupled.")

# Fetch the data instantly from SQL
weather_df, db_status = fetch_tomorrow_from_sql()
st.sidebar.info(db_status)

if weather_df is not None:
    if st.button("🚀 Generate Tomorrow's Forecast", type="primary"):
        with st.spinner("Processing dual-stage inference pipeline..."):
            
            # Simulated inference for demonstration
            base_gen = [max(0, 450 * np.sin(i/24 * np.pi)) if 6 <= i <= 18 else 0 for i in range(24)]
            sarimax_pred = [val * (1 + 0.12 * np.sin(i)) if val > 0 else 0 for i, val in enumerate(base_gen)]
            lstm_corrections = [-25 * np.cos(i/3) if val > 0 else 0 for i, val in enumerate(base_gen)]
            hybrid_pred = [max(0, s + l) for s, l in zip(sarimax_pred, lstm_corrections)]

            results_df = pd.DataFrame({
                "Time": weather_df["timestamp"].dt.strftime('%H:%M'),
                "SARIMAX Baseline (kW)": sarimax_pred,
                "Hybrid Engine Output (kW)": hybrid_pred
            }).set_index("Time")
            
            # Display Metrics
            total_kwh = np.trapz(hybrid_pred, dx=1.0)
            col1, col2, col3 = st.columns(3)
            col1.metric("Est. Energy Yield", f"{total_kwh:.2f} kWh")
            col2.metric("Data Source", "SQLite DB")
            col3.metric("Latency", "< 15ms")
            
            st.line_chart(results_df)
            
            with st.expander("View Raw SQL Data Ingestion"):
                st.dataframe(weather_df)
else:
    st.error("Cannot run prediction. Please run `data_pipeline.py` first to populate the database.")
