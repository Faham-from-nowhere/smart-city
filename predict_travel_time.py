import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import os

# --- Configuration ---
MODEL_PATH = "traffic_prediction_model.pkl"
PROCESSED_DATA_FOR_COLUMNS = "processed_combined_data_hyd.csv" # Used to get the exact column order and names

# This needs to match the FETCH_INTERVAL_MINUTES used during preprocessing
# for consistent timestamp rounding.
FETCH_INTERVAL_MINUTES = 5

def load_model_and_features(model_path, processed_data_path):
    """
    Loads the trained model and the feature column names used during training.
    """
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    print("Model loaded successfully.")

    print(f"Loading feature columns from {processed_data_path} for consistency...")
    # Load the processed data to get the exact columns used for training
    df_processed = pd.read_csv(processed_data_path)

    # Re-create the exact features_to_exclude list used during training
    # Ensure this matches train_model.py's features_to_exclude
    features_to_exclude_from_raw = [
        'timestamp',              # Not a direct feature, temporal aspects are extracted
        'travel_time_seconds',    # Target is travel_time_minutes, seconds is redundant
        'distance_meters',        # Redundant if distance_km is used, or can be included if desired
        'speed_kmph',             # Excluded during training
        'travel_time_minutes'     # This is our target variable (y)
    ]
    # Get the feature columns that were actually used for training
    # This ensures consistency even if columns were dropped/added.
    feature_columns = df_processed.drop(columns=features_to_exclude_from_raw, errors='ignore').columns.tolist()
    print("Feature columns successfully loaded.")

    return model, feature_columns


# MODIFIED: Added 'processed_data_path_ref' argument
def preprocess_new_data(input_data, feature_columns, fetch_interval_minutes, processed_data_path_ref):
    """
    Preprocesses new single input data point to match the model's expected feature format.
    Args:
        input_data (dict): A dictionary containing raw input for prediction, e.g.:
                           {'origin': 'Secunderabad Railway Station, Secunderabad, Telangana',
                            'destination': 'Hitech City, Hyderabad, Telangana',
                            'current_datetime': datetime_object, # Python datetime object
                            'distance_km': 15.0, # Approximate distance for the route
                            'current_aqi_value': 120,
                            'current_pm25': 70,
                            'current_pm10': 100,
                            'current_o3': 40,
                            'current_co': 10,
                            'current_so2': 5,
                            'current_no2': 20,
                            'summary_route': 'NH 65' # Important for one-hot encoding
                           }
        feature_columns (list): The ordered list of feature column names the model was trained on.
        fetch_interval_minutes (int): Interval used for rounding timestamps during preprocessing.
        processed_data_path_ref (str): Path to the processed data file, used to get all unique categorical values.
    Returns:
        pd.DataFrame: A single-row DataFrame ready for model prediction.
    """
    # Create a DataFrame from the single input data point
    df_new = pd.DataFrame([input_data])

    # 1. Temporal features (from current_datetime)
    df_new['timestamp'] = df_new['current_datetime']
    df_new['hour_of_day'] = df_new['timestamp'].dt.hour
    df_new['day_of_week'] = df_new['timestamp'].dt.dayofweek
    df_new['day_of_month'] = df_new['timestamp'].dt.day
    df_new['month'] = df_new['timestamp'].dt.month
    df_new['year'] = df_new['timestamp'].dt.year
    df_new['is_weekend'] = df_new['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    df_new['is_morning_rush_hour'] = df_new['hour_of_day'].apply(lambda x: 1 if 8 <= x <= 10 else 0)
    df_new['is_evening_rush_hour'] = df_new['hour_of_day'].apply(lambda x: 1 if 17 <= x <= 20 else 0)
    df_new['is_rush_hour'] = ((df_new['is_morning_rush_hour'] == 1) | (df_new['is_evening_rush_hour'] == 1)).astype(int)

    # 2. Round timestamp for pollution alignment (even if we only have one point now)
    # FutureWarning: 'T' is deprecated and will be removed in a future version, please use 'min' instead.
    # To fix this warning, change f'{fetch_interval_minutes}T' to f'{fetch_interval_minutes}min'
    df_new['rounded_timestamp'] = df_new['timestamp'].dt.round(f'{fetch_interval_minutes}T')


    # Rename current pollution values to match aggregated column names used in training
    df_new['avg_aqi_value'] = df_new['current_aqi_value']
    # Explicitly assign other pollution columns to ensure they are present if they were in training data
    # and were included in input_data
    df_new['pm25'] = df_new['current_pm25']
    df_new['pm10'] = df_new['current_pm10']
    df_new['o3'] = df_new['current_o3']
    df_new['co'] = df_new['current_co']
    df_new['so2'] = df_new['current_so2']
    df_new['no2'] = df_new['current_no2']


    # 3. Handle Categorical Features (One-Hot Encoding)
    # Load the processed data reference to get all unique categorical values and column order
    # MODIFIED: Using 'processed_data_path_ref' argument
    df_train_columns_ref = pd.read_csv(processed_data_path_ref)

    # Get the original categorical columns that were encoded
    # IMPORTANT: These must match the original string names in your raw data
    original_categorical_cols = ['origin', 'destination', 'summary_route']

    # Create a temporary DataFrame for one-hot encoding the *new* input
    temp_df_categorical = pd.DataFrame({
        'origin': [input_data['origin']],
        'destination': [input_data['destination']],
        'summary_route': [input_data['summary_route']]
    })

    # Get dummies for the input data (will only have columns for the specific input values)
    df_new_encoded = pd.get_dummies(temp_df_categorical, columns=original_categorical_cols, drop_first=True)

    # Extract only the one-hot encoded column names from the training data reference
    training_ohe_cols = [col for col in feature_columns if col.startswith(tuple([f'{c}_' for c in original_categorical_cols]))]

    # Initialize all training_ohe_cols in df_new to 0
    for col in training_ohe_cols:
        df_new[col] = 0

    # Populate df_new with the specific one-hot encoded values for the current input
    for col in df_new_encoded.columns:
        if col in df_new.columns: # Ensure the column exists in our target df_new
            df_new[col] = df_new_encoded[col].iloc[0]


    # Drop the original categorical columns and temporary timestamp/pollution columns used for processing
    df_new = df_new.drop(columns=[
        'origin', 'destination', 'summary_route', 'timestamp',
        'current_datetime', 'rounded_timestamp', 'current_aqi_value',
        'current_pm25', 'current_pm10', 'current_o3', 'current_co', 'current_so2', 'current_no2'
    ], errors='ignore')

    # Ensure the final DataFrame has exactly the same columns in the same order as feature_columns
    # This is CRITICAL for model compatibility.
    # Create an empty DataFrame with the exact target columns and then populate it.
    final_df_for_prediction = pd.DataFrame(columns=feature_columns)
    
    # Iterate through each required feature column
    row_data = {}
    for col in feature_columns:
        if col in df_new.columns:
            row_data[col] = df_new[col].iloc[0]
        else:
            row_data[col] = 0 # Default to 0 if the feature was not generated (e.g., specific OHE column not activated)
    
    final_df_for_prediction = pd.DataFrame([row_data])

    # Correct boolean columns to integer 0/1 if they are still bool (pd.get_dummies sometimes returns bool)
    for col in final_df_for_prediction.columns:
        if final_df_for_prediction[col].dtype == bool:
            final_df_for_prediction[col] = final_df_for_prediction[col].astype(int)

    return final_df_for_prediction


def predict_travel_time(model, feature_columns, processed_data_path_ref, input_data):
    """
    Predicts travel time for a single input data point.
    """
    # Preprocess the new data
    # MODIFIED: Pass 'processed_data_path_ref' here
    processed_input = preprocess_new_data(input_data, feature_columns, FETCH_INTERVAL_MINUTES, processed_data_path_ref)

    if processed_input.empty:
        print("Error: Preprocessing resulted in empty data. Cannot predict.")
        return None
    
    # Make prediction
    predicted_time = model.predict(processed_input)[0]
    return predicted_time

if __name__ == "__main__":
    # Load model and feature columns once
    trained_model, training_feature_columns = load_model_and_features(MODEL_PATH, PROCESSED_DATA_FOR_COLUMNS)

    if trained_model is None or training_feature_columns is None:
        print("Failed to load model or feature columns. Exiting.")
        exit()

    # --- Example Usage ---
    print("\n--- Testing Prediction ---")
    
    # You will replace these with real-time incoming data in your FastAPI endpoint
    example_input = {
        'origin': 'Secunderabad Railway Station, Secunderabad, Telangana',
        'destination': 'Hitech City, Hyderabad, Telangana',
        'current_datetime': datetime.now(), # Use current system time
        'distance_km': 15.0, # Approximate distance for this route. In a real system, you'd get this from Maps API
        'current_aqi_value': 80, # Example AQI value
        'current_pm25': 45,
        'current_pm10': 70,
        'current_o3': 30,
        'current_co': 8,
        'current_so2': 3,
        'current_no2': 15,
        'summary_route': 'Kukatpally Housing Board - Hitech City Rd' # Make sure this matches your summary_route_ columns
    }

    print(f"Input for prediction: {example_input['origin']} to {example_input['destination']} at {example_input['current_datetime'].strftime('%Y-%m-%d %H:%M')}, AQI: {example_input['current_aqi_value']}")
    
    # MODIFIED: Pass 'PROCESSED_DATA_FOR_COLUMNS' to predict_travel_time
    predicted_travel_time = predict_travel_time(trained_model, training_feature_columns, PROCESSED_DATA_FOR_COLUMNS, example_input)

    if predicted_travel_time is not None:
        print(f"Predicted travel time: {predicted_travel_time:.2f} minutes")

    print("\n--- Another Example ---")
    example_input_2 = {
        'origin': 'Jubilee Hills Check Post, Hyderabad, Telangana',
        'destination': 'Banjara Hills Road No 12, Hyderabad, Telangana',
        'current_datetime': datetime(2025, 6, 4, 18, 0, 0), # Tomorrow 6 PM (evening rush hour)
        'distance_km': 3.5,
        'current_aqi_value': 150, # Higher pollution example
        'current_pm25': 90,
        'current_pm10': 120,
        'current_o3': 50,
        'current_co': 15,
        'current_so2': 7,
        'current_no2': 25,
        'summary_route': 'Road No. 36' # Example route
    }
    
    print(f"Input for prediction: {example_input_2['origin']} to {example_input_2['destination']} at {example_input_2['current_datetime'].strftime('%Y-%m-%d %H:%M')}, AQI: {example_input_2['current_aqi_value']}")
    
    # MODIFIED: Pass 'PROCESSED_DATA_FOR_COLUMNS' to predict_travel_time
    predicted_travel_time_2 = predict_travel_time(trained_model, training_feature_columns, PROCESSED_DATA_FOR_COLUMNS, example_input_2)

    if predicted_travel_time_2 is not None:
        print(f"Predicted travel time: {predicted_travel_time_2:.2f} minutes")