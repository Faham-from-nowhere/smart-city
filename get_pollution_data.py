import requests
import pandas as pd
from datetime import datetime
import time
import os

# --- Configuration ---
YOUR_AQICN_API_TOKEN = "d0b68b5b03b0dc82f37d01b0a9ca1c5d07dfdf63" # <--- REPLACE WITH YOUR ACTUAL TOKEN

# Define AQI monitoring stations in Hyderabad (You'll need to verify these codes/names via AQICN map/API)
# For specific stations, you might use their UID (e.g., "@1234") or name like "Delhi-ITO"
# For a broader city average, you can sometimes use the city name directly as a feed.
# Check https://aqicn.org/map/india/hyderabad/ and click on stations to see their names or UIDs.
AQI_STATIONS_TO_MONITOR = [
    "Hyderabad", # This typically pulls from a major station in Hyderabad
    "@H14156", # Example: Hyderabad US Consulate (UID found from AQICN map) - Verify this UID!
    "@H14149", # Example station name
    "@H11284",
    "@H14140",
]

POLLUTION_OUTPUT_FILE = "pollution_data_hyd.csv"
FETCH_INTERVAL_MINUTES = 5 # Fetch pollution data every 5 minutes

def get_aqi_data_for_station(station_identifier, api_token):
    """
    Fetches real-time AQI data for a given station using AQICN API.
    `station_identifier` can be a city name or a specific station UID/name.
    """
    try:
        url = f"https://api.waqi.info/feed/{station_identifier}/?token={api_token}"

        response = requests.get(url, timeout=10) # Added timeout
        response.raise_for_status() # Raise an exception for HTTP errors
        aqi_json = response.json()

        if aqi_json and aqi_json['status'] == 'ok':
            data = aqi_json['data']
            aqi_value = data.get('aqi')
            iaqi_data = data.get('iaqi', {}) # Individual pollutant values

            # Extract common pollutants if available, default to None if not present
            pm25 = iaqi_data.get('pm25', {}).get('v')
            pm10 = iaqi_data.get('pm10', {}).get('v')
            o3 = iaqi_data.get('o3', {}).get('v')
            co = iaqi_data.get('co', {}).get('v')
            so2 = iaqi_data.get('so2', {}).get('v')
            no2 = iaqi_data.get('no2', {}).get('v')

            return {
                "timestamp": datetime.now().isoformat(),
                "aqi_station": station_identifier,
                "aqi_value": aqi_value,
                "pm25": pm25,
                "pm10": pm10,
                "o3": o3,
                "co": co,
                "so2": so2,
                "no2": no2,
                "api_source": "AQICN" # To track source
            }
        else:
            print(f"AQICN API Error or no data for station {station_identifier}: {aqi_json.get('data', {}).get('message', 'Unknown error')}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"HTTP Request Error for AQICN station {station_identifier}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while fetching AQI for {station_identifier}: {e}")
        return None

if __name__ == "__main__":
    print(f"Starting real-time pollution data collection for {len(AQI_STATIONS_TO_MONITOR)} stations...")

    file_exists = os.path.exists(POLLUTION_OUTPUT_FILE)
    write_header = not file_exists

    while True: # Run continuously
        current_batch_pollution_data = []
        print(f"\n--- Fetching pollution data at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")

        for station in AQI_STATIONS_TO_MONITOR:
            aqi_data = get_aqi_data_for_station(station, YOUR_AQICN_API_TOKEN)
            if aqi_data:
                current_batch_pollution_data.append(aqi_data)
                print(f"  Pollution: {aqi_data['aqi_station']} - AQI: {aqi_data['aqi_value']}")
            else:
                print(f"  Pollution Failed for: {station}")

        if current_batch_pollution_data:
            df_pollution_batch = pd.DataFrame(current_batch_pollution_data)
            df_pollution_batch.to_csv(POLLUTION_OUTPUT_FILE, mode='a', header=write_header, index=False)
            write_header = False
            print(f"Successfully appended {len(current_batch_pollution_data)} pollution data points to {POLLUTION_OUTPUT_FILE}")
        else:
            print("No pollution data collected in this batch.")

        print(f"Waiting for {FETCH_INTERVAL_MINUTES} minutes before next pollution fetch cycle...")
        time.sleep(FETCH_INTERVAL_MINUTES * 60)