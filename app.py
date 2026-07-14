import os
import tempfile

# Force the database file to sit in a cloud-writable scratchpad directory
DB_PATH = os.path.join(tempfile.gettempdir(), "solar_data_fleet.db")

# -----------------------------------------------------------------------------
# MULTI-LOCATION SELF-HEALING DATABASE FILLER
# -----------------------------------------------------------------------------
def initialize_and_populate_db(location_name, lat, lon):
    """Creates tables with a location key and ensures data exists for the selected city."""
    conn = sqlite3.connect(DB_PATH)  # <-- Changed to safe path
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS real_time_weather (
            location TEXT,
            timestamp DATETIME,
            temperature REAL,
            humidity REAL,
            cloud_cover REAL,
            irradiance REAL,
            PRIMARY KEY (location, timestamp)
        )
    ''')
    conn.commit()
    
    tomorrow_str = str(date.today() + timedelta(days=1))
    cursor.execute(
        "SELECT COUNT(*) FROM real_time_weather WHERE location = ? AND timestamp LIKE ?", 
        (location_name, f"{tomorrow_str}%")
    )
    count = cursor.fetchone()[0]
    
    if count == 0:
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

# -----------------------------------------------------------------------------
# CORE DATA RETRIEVAL LAYER
# -----------------------------------------------------------------------------
def fetch_location_data_from_sql(location_name):
    try:
        conn = sqlite3.connect(DB_PATH)  # <-- Changed to safe path
        tomorrow_str = str(date.today() + timedelta(days=1))
        
        query = f"""
            SELECT timestamp, temperature, humidity, cloud_cover, irradiance
            FROM real_time_weather 
            WHERE location = '{location_name}' AND timestamp LIKE '{tomorrow_str}%'
            ORDER BY timestamp ASC
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if df.empty:
            return None, "⚠️ Syncing telemetry stream from Open-Meteo API..."
            
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df, f"✅ SQL Sync Active: {location_name} Node Online"
        
    except Exception as e:
        return None, f"Database Error: {str(e)}"
