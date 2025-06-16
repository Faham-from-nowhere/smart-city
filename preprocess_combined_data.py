import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# --- Configuration ---
TRAFFIC_INPUT_FILE = "all_traffic_data_hyd.csv"
POLLUTION_INPUT_FILE = "pollution_data_hyd.csv"
PROCESSED_DATA_OUTPUT_FILE = "processed_combined_data_hyd.csv"

# Make sure this matches the actual fetch interval you used in your data collection scripts
# (e.g., if you fetched every 5 minutes, set this to 5)
FETCH_INTERVAL_MINUTES = 5

def preprocess_combined_data(df_traffic, df_pollution):
    """
    Performs preprocessing and feature engineering on combined traffic and pollution data.
    """
    print("Starting data preprocessing and feature engineering...")

    # --- Process Traffic Data ---
    # Robust timestamp conversion for traffic data
    df_traffic['timestamp'] = pd.to_datetime(df_traffic['timestamp'], errors='coerce')
    df_traffic.dropna(subset=['timestamp'], inplace=True) # Drop rows where timestamp couldn't be parsed
    df_traffic = df_traffic.sort_values('timestamp').reset_index(drop=True)

    # 1. Extract Temporal features for traffic
    df_traffic['hour_of_day'] = df_traffic['timestamp'].dt.hour
    df_traffic['day_of_week'] = df_traffic['timestamp'].dt.dayofweek # Monday=0, Sunday=6
    df_traffic['day_of_month'] = df_traffic['timestamp'].dt.day
    df_traffic['month'] = df_traffic['timestamp'].dt.month
    df_traffic['year'] = df_traffic['timestamp'].dt.year

    # 2. Derive 'Is_Weekend' and 'Is_Rush_Hour'
    df_traffic['is_weekend'] = df_traffic['day_of_week'].apply(lambda x: 1 if x >= 5 else 0) # Saturday (5) and Sunday (6)
    df_traffic['is_morning_rush_hour'] = df_traffic['hour_of_day'].apply(lambda x: 1 if 8 <= x <= 10 else 0)
    df_traffic['is_evening_rush_hour'] = df_traffic['hour_of_day'].apply(lambda x: 1 if 17 <= x <= 20 else 0)
    df_traffic['is_rush_hour'] = ((df_traffic['is_morning_rush_hour'] == 1) | (df_traffic['is_evening_rush_hour'] == 1)).astype(int)

    # 3. Calculate Speed (km/h) for traffic
    df_traffic['speed_kmph'] = df_traffic.apply(
        lambda row: (row['distance_meters'] / row['travel_time_seconds']) * 3.6
        if pd.notna(row['travel_time_seconds']) and row['travel_time_seconds'] > 0 else np.nan, axis=1
    )

    # --- Process Pollution Data ---
    # Robust timestamp conversion for pollution data
    df_pollution['timestamp'] = pd.to_datetime(df_pollution['timestamp'], errors='coerce')
    df_pollution.dropna(subset=['timestamp'], inplace=True) # Drop rows where timestamp couldn't be parsed
    df_pollution = df_pollution.sort_values('timestamp').reset_index(drop=True)

    # Convert pollution columns to numeric, coercing non-numeric values to NaN
    pollution_value_cols = ['aqi_value', 'pm25', 'pm10', 'o3', 'co', 'so2', 'no2']
    for col in pollution_value_cols:
        df_pollution[col] = pd.to_numeric(df_pollution[col], errors='coerce')

    # 4. Aggregate Pollution Data to a single AQI per timestamp (Average across stations)
    df_pollution['rounded_timestamp'] = df_pollution['timestamp'].dt.round(f'{FETCH_INTERVAL_MINUTES}T')

    df_pollution_agg = df_pollution.groupby('rounded_timestamp').agg({
        'aqi_value': 'mean',
        'pm25': 'mean',
        'pm10': 'mean',
        'o3': 'mean',
        'co': 'mean',
        'so2': 'mean',
        'no2': 'mean'
    }).reset_index()
    df_pollution_agg = df_pollution_agg.rename(columns={'aqi_value': 'avg_aqi_value'})

    # --- Combine Traffic and Pollution Data ---
    df_traffic['rounded_timestamp'] = df_traffic['timestamp'].dt.round(f'{FETCH_INTERVAL_MINUTES}T')

    df_combined = pd.merge_asof(
        df_traffic.sort_values('rounded_timestamp'),
        df_pollution_agg.sort_values('rounded_timestamp'),
        on='rounded_timestamp',
        direction='nearest',
        tolerance=pd.Timedelta(f'{FETCH_INTERVAL_MINUTES} minutes')
    )
    df_combined = df_combined.drop(columns=['rounded_timestamp'])

    # Fill any remaining NaNs in pollution columns (e.g., if no pollution data for that time)
    # Using forward fill, then filling any initial NaNs (if no prior data) with 0
    for col in pollution_value_cols: # Use the original list of pollution value columns
        # The aggregated column is 'avg_aqi_value', so handle it specifically
        if col == 'aqi_value':
             df_combined['avg_aqi_value'] = df_combined['avg_aqi_value'].fillna(method='ffill').fillna(0)
        else:
            df_combined[col] = df_combined[col].fillna(method='ffill').fillna(0)


    # 5. Categorical Feature Encoding for Traffic-related columns
    df_combined = pd.get_dummies(df_combined, columns=['origin', 'destination', 'summary_route'], drop_first=True)

    print("Data preprocessing and feature engineering complete.")
    return df_combined

if __name__ == "__main__":
    try:
        # Load traffic data
        if not os.path.exists(TRAFFIC_INPUT_FILE):
            print(f"Error: Traffic input file '{TRAFFIC_INPUT_FILE}' not found. Please ensure it exists.")
            exit()
        df_traffic_raw = pd.read_csv(TRAFFIC_INPUT_FILE)
        print(f"Loaded {len(df_traffic_raw)} raw traffic data points from {TRAFFIC_INPUT_FILE}")

        # Define column names for the pollution CSV based on the sample data provided
        pollution_column_names = [
            'timestamp', 'aqi_station', 'aqi_value', 'pm25', 'pm10',
            'o3', 'co', 'so2', 'no2', 'api_source'
        ]

        # Load pollution data, explicitly telling pandas the column names and that there's no header
        if not os.path.exists(POLLUTION_INPUT_FILE):
            print(f"Error: Pollution input file '{POLLUTION_INPUT_FILE}' not found. Please ensure it exists.")
            exit()
        df_pollution_raw = pd.read_csv(POLLUTION_INPUT_FILE, header=None, names=pollution_column_names)
        print(f"Loaded {len(df_pollution_raw)} raw pollution data points from {POLLUTION_INPUT_FILE}")

        # Perform combined preprocessing
        df_processed_combined = preprocess_combined_data(df_traffic_raw.copy(), df_pollution_raw.copy())

        # Display the first few rows of the processed data and its shape
        print("\n--- Processed Combined Data Head (first 5 rows) ---")
        # pd.set_option('display.max_columns', None) # Uncomment to see all columns
        print(df_processed_combined.head())
        print(f"\nShape of processed combined data: {df_processed_combined.shape}")

        # Save the processed data
        df_processed_combined.to_csv(PROCESSED_DATA_OUTPUT_FILE, index=False)
        print(f"\nProcessed combined data saved to {PROCESSED_DATA_OUTPUT_FILE}")

    except Exception as e:
        print(f"An error occurred during combined preprocessing: {e}")