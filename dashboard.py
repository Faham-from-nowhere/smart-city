# dashboard.py
import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import os
import json
import folium
from streamlit_folium import st_folium



AQICN_API_TOKEN = os.getenv("AQICN_API_TOKEN") 
AQICN_BASE_URL = "https://api.waqi.info/feed/"
FASTAPI_URL = "https://smart-city-1h92.onrender.com/predict_travel_time"
PROCESSED_DATA_PATH = "processed_combined_data_hyd.csv" 
RAW_TRAFFIC_DATA_PATH = "all_traffic_data_hyd.csv"


st.set_page_config(page_title="Smart Traffic Flow Optimizer", layout="centered")

# Hyderabad-specific coordinates for origin/destination mapping 
LOCATION_COORDINATES = {
    "Gachibowli, Hyderabad, Telangana": (17.4437, 78.3772),
    "Jubilee Hills Check Post, Hyderabad, Telangana": (17.4334, 78.4027),
    "Koti, Hyderabad, Telangana": (17.3871, 78.4754),
    "LB Nagar, Hyderabad, Telangana": (17.3468, 78.5487),
    "MGBS, Hyderabad, Telangana": (17.3807, 78.4727),
    "Miyapur, Hyderabad, Telangana": (17.4947, 78.3842),
    "Paradise Circle, Secunderabad, Telangana": (17.4385, 78.5034),
    "Secunderabad Railway Station, Malkajgiri, Secunderabad": (17.4399, 78.5003),
    "Secunderabad Railway Station, Secunderabad, Telangana": (17.4399, 78.5003),
    "Banjara Hills Road No 12, Hyderabad, Telangana": (17.4206, 78.4363),
    "Begum Bazaar, Hyderabad, Telangana": (17.3831, 78.4713),
    "Dilsukhnagar, Hyderabad, Telangana": (17.3732, 78.5284),
    "ECIL X Road, Hyderabad, Telangana": (17.4729, 78.5630),
    "Financial District, Hyderabad, Telangana": (17.4442, 78.3698),
    "Hitech City, Hyderabad, Telangana": (17.4475, 78.3768),
    "Keesara, Hyderabad, Telangana": (17.5312, 78.6508),
    "Kukatpally Housing Board, Hyderabad, Telangana": (17.4859, 78.4063),
    "MG Road, Secunderabad, Telangana": (17.4394, 78.4975),
}

# Helper function to fetch AQI data for a specific city/station
@st.cache_data(ttl=3600) # Cache pollution data for 1 hour to avoid excessive API calls
def get_current_pollution_data(station_id_or_city: str, api_token: str):
    """
    Fetches current pollution data (AQI, PM2.5, PM10, etc.) for a given city/station ID.
    Args:
        station_id_or_city (str): AQICN station ID (e.g., "@5888") or city name (e.g., "hyderabad").
        api_token (str): Your AQICN API token.
    Returns:
        dict: A dictionary of pollution values, or None if data fetching fails.
    """
    url = f"{AQICN_BASE_URL}{station_id_or_city}/?token={api_token}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status() 
        data = response.json()

        if data and data['status'] == 'ok':
            iaqi = data['data']['iaqi']
            pollution_values = {
                'current_aqi_value': iaqi.get('aqi', {}).get('v', 0),
                'current_pm25': iaqi.get('pm25', {}).get('v', 0),
                'current_pm10': iaqi.get('pm10', {}).get('v', 0),
                'current_o3': iaqi.get('o3', {}).get('v', 0),
                'current_co': iaqi.get('co', {}).get('v', 0),
                'current_so2': iaqi.get('so2', {}).get('v', 0),
                'current_no2': iaqi.get('no2', {}).get('v', 0)
            }
            return pollution_values
        else:
            print(f"AQICN API returned status: {data.get('status', 'unknown')}. Message: {data.get('data', {}).get('message', 'No message.')}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching pollution data: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during pollution data processing: {e}")
        return None


#  Load data for dropdowns and historical data 
@st.cache_data # Cache this dataframe loading
def load_data_for_dashboard(processed_path, raw_traffic_path):
    # Load processed data (for historical trends and features)
    if not os.path.exists(processed_path):
        print(f"ERROR: Processed data file not found at {processed_path}. Cannot load data for dashboard.")
        return pd.DataFrame(), [], [], []
    df_processed = pd.read_csv(processed_path)
    df_processed['timestamp'] = pd.to_datetime(df_processed['timestamp'])

    # Load raw traffic data (to get original origin, destination, summary_route strings)
    if not os.path.exists(raw_traffic_path):
        print(f"ERROR: Raw traffic data file not found at {raw_traffic_path}. Cannot populate dropdowns correctly.")
        # Fallback: try to infer from processed data (less reliable)
        unique_origins = sorted(list(set([col.replace('origin_', '') for col in df_processed.columns if col.startswith('origin_')])))
        unique_destinations = sorted(list(set([col.replace('destination_', '') for col in df_processed.columns if col.startswith('destination_')])))
        unique_summary_routes = sorted(list(set([col.replace('summary_route_', '') for col in df_processed.columns if col.startswith('summary_route_')])))
        return df_processed, unique_origins, unique_destinations, unique_summary_routes

    df_raw_traffic = pd.read_csv(raw_traffic_path)
    df_raw_traffic['timestamp'] = pd.to_datetime(df_raw_traffic['timestamp'])

   
    # Use merge_asof for robust time-based merging if timestamps don't exactly align
    # Ensure both dataframes are sorted by timestamp before merge_asof
    df_merged = pd.merge_asof(
        df_processed.sort_values('timestamp'),
        df_raw_traffic[['timestamp', 'origin', 'destination', 'summary_route']].sort_values('timestamp'),
        on='timestamp',
        direction='nearest',
        tolerance=pd.Timedelta('1 minute') # Allow small differences for merging
    )

    # Clean up merged data (e.g., drop rows where merge failed)
    df_merged.dropna(subset=['origin', 'destination', 'summary_route'], inplace=True)

    # Get unique values for dropdowns from the merged dataframe
    unique_origins = sorted(df_merged['origin'].unique().tolist())
    unique_destinations = sorted(df_merged['destination'].unique().tolist())
    unique_summary_routes = sorted(df_merged['summary_route'].unique().tolist())

    # Filter out empty strings if any
    unique_origins = [o for o in unique_origins if o]
    unique_destinations = [d for d in unique_destinations if d]
    unique_summary_routes = [s for s in unique_summary_routes if s]

    return df_merged, unique_origins, unique_destinations, unique_summary_routes

# Load unique values and the full processed dataframe once
full_processed_df, origins, destinations, summary_routes = load_data_for_dashboard(PROCESSED_DATA_PATH, RAW_TRAFFIC_DATA_PATH)


#  Streamlit UI 
st.title("🚦 Smart Traffic Flow Optimizer")
st.markdown("Predict travel time based on route details, time of day, and environmental factors.")

# Input fields for traffic details
st.header("Route Details")
col1, col2 = st.columns(2)
with col1:
    selected_origin = st.selectbox("Origin", options=origins, help="Starting point of your journey.")
with col2:
    selected_destination = st.selectbox("Destination", options=destinations, help="End point of your journey.")

selected_summary_route = st.selectbox("Summary Route (Important for accuracy)", options=summary_routes, help="The main road or highway connecting origin to destination. Must match training data.")
approx_distance_km = st.number_input("Approximate Distance (km)", min_value=0.1, max_value=200.0, value=10.0, step=0.1, help="Estimated distance of the chosen route in kilometers.")

# --- Map Visualization ---
st.subheader("Route Overview Map")
origin_coords = LOCATION_COORDINATES.get(selected_origin)
destination_coords = LOCATION_COORDINATES.get(selected_destination)

if origin_coords and destination_coords:
    center_lat = (origin_coords[0] + destination_coords[0]) / 2
    center_lon = (origin_coords[1] + destination_coords[1]) / 2

    m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

    folium.Marker(origin_coords, popup=f"**Origin:** {selected_origin}", icon=folium.Icon(color='green', icon='play')).add_to(m)
    folium.Marker(destination_coords, popup=f"**Destination:** {selected_destination}", icon=folium.Icon(color='red', icon='stop')).add_to(m)

    folium.PolyLine([origin_coords, destination_coords], color="blue", weight=2.5, opacity=1).add_to(m)

    st_folium(m, width=700, height=350, returned_objects=[])

else:
    st.info("Select valid Origin and Destination to display on the map.")


st.header("Environmental Conditions (AQI)")
aqi_station = st.text_input("AQI Monitoring Station/City", value="hyderabad", help="Enter a city name or AQICN station ID (e.g., 'hyderabad', '@H14156') to fetch pollution data.")
fetch_pollution = st.button("Fetch Current Pollution Data")

current_pollution_data = None
if fetch_pollution:
    if AQICN_API_TOKEN == "YOUR_AQICN_API_TOKEN":
        st.error("Please replace 'YOUR_AQICN_API_TOKEN' with your actual AQICN API token in dashboard.py!")
        st.info("Using default/zero pollution values for prediction.")
        current_pollution_data = {
            'current_aqi_value': 0, 'current_pm25': 0, 'current_pm10': 0,
            'current_o3': 0, 'current_co': 0, 'current_so2': 0, 'current_no2': 0
        }
    else:
        with st.spinner("Fetching pollution data... (Requires Internet)"):
            current_pollution_data = get_current_pollution_data(aqi_station, AQICN_API_TOKEN)
        if current_pollution_data:
            st.success("Pollution data fetched successfully!")
            st.json(current_pollution_data)
        else:
            st.error("Could not fetch pollution data. Using default/zero values for prediction.")
            current_pollution_data = {
                'current_aqi_value': 0, 'current_pm25': 0, 'current_pm10': 0,
                'current_o3': 0, 'current_co': 0, 'current_so2': 0, 'current_no2': 0
            }
else:
    current_pollution_data = {
        'current_aqi_value': 0, 'current_pm25': 0, 'current_pm10': 0,
        'current_o3': 0, 'current_co': 0, 'current_so2': 0, 'current_no2': 0
    }

st.markdown("---")

if st.button("Predict Travel Time"):
    if not selected_origin or not selected_destination or not selected_summary_route:
        st.warning("Please select Origin, Destination, and Summary Route.")
    else:
        try:
            request_payload = {
                "origin": selected_origin,
                "destination": selected_destination,
                "distance_km": approx_distance_km,
                "current_aqi_value": current_pollution_data['current_aqi_value'],
                "current_pm25": current_pollution_data['current_pm25'],
                "current_pm10": current_pollution_data['current_pm10'],
                "current_o3": current_pollution_data['current_o3'],
                "current_co": current_pollution_data['current_co'],
                "current_so2": current_pollution_data['current_so2'],
                "current_no2": current_pollution_data['current_no2'],
                "summary_route": selected_summary_route
            }

            with st.spinner("Getting prediction from API..."):
                response = requests.post(FASTAPI_URL, json=request_payload, timeout=20)
                response.raise_for_status()
                prediction_result = response.json()

            predicted_minutes = prediction_result.get("predicted_travel_time_minutes")
            
            if predicted_minutes is not None:
                st.success(f"**Predicted Travel Time: {predicted_minutes:.2f} minutes**")
            else:
                st.error("Prediction failed: Could not retrieve predicted time from API response.")
                st.json(prediction_result)

        except requests.exceptions.ConnectionError:
            st.error(f"Could not connect to FastAPI service at {FASTAPI_URL}. Is it running?")
            st.info("Please ensure your FastAPI service is running in a separate terminal using: `uvicorn api_service:app --host 0.0.0.0 --port 8000 --reload`")
        except requests.exceptions.Timeout:
            st.error("Request to FastAPI service timed out. The service might be slow or unresponsive.")
        except requests.exceptions.RequestException as e:
            st.error(f"Error communicating with FastAPI service: {e}")
            if response is not None:
                st.error(f"API Response: {response.text}")
        except Exception as e:
            st.error(f"An unexpected error occurred during prediction: {e}")

st.markdown("---")

# --- Historical Trends Section ---
st.header("Historical Travel Time Trends")
st.info("Select a Route to View Historical Data.")

if not full_processed_df.empty:
    # Use the 'origin', 'destination', 'summary_route' columns from the merged dataframe
    # These columns now exist because of the merge with raw_traffic_data
    full_processed_df['route_identifier'] = full_processed_df['origin'] + " to " + full_processed_df['destination'] + " via " + full_processed_df['summary_route']
    
    unique_route_identifiers = sorted(full_processed_df['route_identifier'].unique().tolist())

    selected_historical_route = st.selectbox(
        "Select a Route to View Historical Data",
        options=[''] + unique_route_identifiers
    )

    if selected_historical_route:
        historical_data = full_processed_df[full_processed_df['route_identifier'] == selected_historical_route]
        
        if not historical_data.empty:
            historical_data = historical_data.sort_values('timestamp')

            st.subheader(f"Historical Data for: {selected_historical_route}")
            st.write(f"Number of historical records: {len(historical_data)}")
            st.write(f"Average travel time: {historical_data['travel_time_minutes'].mean():.2f} minutes")
            
            st.line_chart(historical_data[['timestamp', 'travel_time_minutes']].set_index('timestamp'))
            
            with st.expander("Show Raw Historical Data"):
                st.dataframe(historical_data[['timestamp', 'travel_time_minutes', 'avg_aqi_value']].head())
        else:
            st.warning("No historical data found for the selected route combination.")
else:
    st.warning("No processed data available to display historical trends. Please ensure 'processed_combined_data_hyd.csv' and 'all_traffic_data_hyd.csv' exist and are populated.")
