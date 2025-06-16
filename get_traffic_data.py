import googlemaps
import pandas as pd
from datetime import datetime
import time
import os # For checking if file exists

# --- Configuration ---
YOUR_API_KEY = os.getenv("YOUR_API_KEY")  # Ensure you set your Google Maps API key in environment variables
gmaps = googlemaps.Client(key=YOUR_API_KEY)

# Define a list of important routes (Origin, Destination)
# You can add more as you identify key areas in Hyderabad/Secunderabad
ROUTES_TO_MONITOR = [
    ("Secunderabad Railway Station, Secunderabad, Telangana", "Hitech City, Hyderabad, Telangana"),
    ("Gachibowli, Hyderabad, Telangana", "Financial District, Hyderabad, Telangana"),
    ("Koti, Hyderabad, Telangana", "Begum Bazaar, Hyderabad, Telangana"),
    ("Jubilee Hills Check Post, Hyderabad, Telangana", "Banjara Hills Road No 12, Hyderabad, Telangana"),
    ("MGBS, Hyderabad, Telangana", "Kukatpally Housing Board, Hyderabad, Telangana"),
    ("LB Nagar, Hyderabad, Telangana", "Dilsukhnagar, Hyderabad, Telangana"),
    ("Miyapur, Hyderabad, Telangana", "Ameerpet, Hyderabad, Telangana"),
    ("Paradise Circle, Secunderabad, Telangana", "MG Road, Secunderabad, Telangana"),
    ("Secunderabad Railway Station, Malkajgiri, Secunderabad", "ECIL X Road, Hyderabad, Telangana"),
    ("ECIL X Road, Hyderabad, Telangana", "Keesara, Hyderabad, Telangana"),

    # Add more routes as needed to cover your target area
]

DATA_OUTPUT_FILE = "all_traffic_data_hyd.csv"
FETCH_INTERVAL_MINUTES = 5 # Fetch data for all routes every 5 minutes

def get_traffic_data_for_route(origin_address, destination_address):
    """
    Fetches real-time traffic data (travel time) for a given route.
    Uses Directions API.
    """
    try:
        directions_result = gmaps.directions(
            origin_address,
            destination_address,
            mode="driving",
            departure_time="now",
            traffic_model="best_guess"
        )

        if directions_result:
            leg = directions_result[0]['legs'][0]
            travel_time_in_traffic = leg['duration_in_traffic']['value'] # in seconds
            distance = leg['distance']['value'] # in meters
            summary = directions_result[0]['summary']
            # start_address = leg['start_address'] # Already in origin_address
            # end_address = leg['end_address'] # Already in destination_address

            data = {
                "timestamp": datetime.now().isoformat(),
                "origin": origin_address,
                "destination": destination_address,
                "summary_route": summary,
                "travel_time_seconds": travel_time_in_traffic,
                "travel_time_minutes": round(travel_time_in_traffic / 60, 2),
                "distance_meters": distance,
                "distance_km": round(distance / 1000, 2),
                # If you need more specific start/end addresses from GMaps, uncomment below
                # "start_full_address": leg['start_address'],
                # "end_full_address": leg['end_address']
            }
            return data
        else:
            print(f"No directions found for {origin_address} to {destination_address}")
            return None

    except googlemaps.exceptions.ApiError as e:
        print(f"Google Maps API Error for {origin_address} to {destination_address}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred for {origin_address} to {destination_address}: {e}")
        return None

if __name__ == "__main__":
    print(f"Starting real-time traffic data collection for {len(ROUTES_TO_MONITOR)} routes...")

    # Check if header is needed for CSV
    file_exists = os.path.exists(DATA_OUTPUT_FILE)
    write_header = not file_exists

    while True: # Run continuously
        current_batch_data = []
        print(f"\n--- Fetching data for all routes at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
        for origin, destination in ROUTES_TO_MONITOR:
            traffic_data = get_traffic_data_for_route(origin, destination)
            if traffic_data:
                current_batch_data.append(traffic_data)
                print(f"  Fetched: {traffic_data['origin']} to {traffic_data['destination']} - Travel Time: {traffic_data['travel_time_minutes']} minutes")
            else:
                print(f"  Failed for: {origin} to {destination}")

        if current_batch_data:
            df_batch = pd.DataFrame(current_batch_data)
            df_batch.to_csv(DATA_OUTPUT_FILE, mode='a', header=write_header, index=False)
            write_header = False # Only write header once
            print(f"Successfully appended {len(current_batch_data)} data points to {DATA_OUTPUT_FILE}")
        else:
            print("No data collected in this batch.")

        print(f"Waiting for {FETCH_INTERVAL_MINUTES} minutes before next fetch cycle...")
        time.sleep(FETCH_INTERVAL_MINUTES * 60)