import sqlite3
import requests
import pandas as pd
from datetime import datetime

def build_database_and_fetch():
    print(f"[{datetime.now()}] Starting automated ETL pipeline...")
    
    # 1. Connect to SQL (This automatically creates solar_data.db)
    conn = sqlite3.connect("solar_data.db")
    cursor = conn.cursor()
    
    # 2. Build the table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS real_time_weather (
            timestamp DATETIME PRIMARY KEY,
            temperature REAL,
            humidity REAL,
            cloud_cover REAL,
            irradiance REAL
        )
    ''')
    
    # 3. Fetch Data from Open-Meteo API (For Chennai/Target Region)
    lat, lon = 13.0827, 80.2707
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_20m,relative_humidity_2m,cloud_cover,direct_radiation&past_days=1&forecast_days=2"
    
    print("Fetching live data from Open-Meteo API...")
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(data["hourly"]["time"]),
            "temperature": data["hourly"]["temperature_20m"],
            "humidity": data["hourly"]["relative_humidity_2m"],
            "cloud_cover": data["hourly"]["cloud_cover"],
            "irradiance": data["hourly"]["direct_radiation"]
        })
        
        # 4. Save to SQL Database
        df.to_sql("real_time_weather", conn, if_exists="append", index=False, method="multi")
        
        # SQL Magic: Delete exact duplicates if script runs twice in one day
        cursor.execute('''
            DELETE FROM real_time_weather 
            WHERE rowid NOT IN (
                SELECT MIN(rowid) FROM real_time_weather GROUP BY timestamp
            )
        ''')
        conn.commit()
        print(f"✅ SUCCESS: Saved {len(df)} hourly records to SQL database 'solar_data.db'")
    else:
        print("❌ API Request Failed.")
        
    conn.close()

if __name__ == "__main__":
    build_database_and_fetch()
