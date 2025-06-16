# api_service.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import pandas as pd
import joblib
import os
from typing import Literal

# --- Configuration (MUST MATCH predict_travel_time.py and train_model.py) ---
MODEL_PATH = "traffic_prediction_model.pkl"
PROCESSED_DATA_FOR_COLUMNS = "processed_combined_data_hyd.csv"
FETCH_INTERVAL_MINUTES = 5 # Ensure this matches your data collection and preprocessing scripts

# --- Global variables to store loaded model and feature columns ---
model = None
feature_columns = None
app = FastAPI(title="Smart Traffic Flow Optimizer API")

# --- Pydantic Model for Request Body ---
# This defines the expected structure and data types of incoming prediction requests
class PredictionRequest(BaseModel):
    origin: str
    destination: str
    # current_datetime will be derived from the API call's time or provided as an option
    # For now, we'll derive it from datetime.now() within the endpoint
    distance_km: float # This would typically come from a Maps API call
    current_aqi_value: float = 0 # Default to 0 if not provided
    current_pm25: float = 0
    current_pm10: float = 0
    current_o3: float = 0
    current_co: float = 0
    current_so2: float = 0
    current_no2: float = 0
    summary_route: str # Crucial for one-hot encoding

# --- Helper Function to Preprocess New Data (Copied from predict_travel_time.py) ---
def preprocess_new_data_for_api(input_data: dict, feature_cols: list, fetch_interval: int, processed_data_ref_path: str):
    """
    Preprocesses new single input data point to match the model's expected feature format.
    Simplified slightly for API context assuming input_data keys are exact.
    """
    df_new = pd.DataFrame([input_data])

    # 1. Temporal features (from current_datetime)
    # Ensure current_datetime is a datetime object
    if not isinstance(df_new['current_datetime'].iloc[0], datetime):
        df_new['current_datetime'] = pd.to_datetime(df_new['current_datetime']) # If passed as string
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

    # 2. Round timestamp for pollution alignment
    # Changed 'T' to 'min' to avoid FutureWarning
    df_new['rounded_timestamp'] = df_new['timestamp'].dt.round(f'{fetch_interval}min')

    # Rename current pollution values to match aggregated column names used in training
    df_new['avg_aqi_value'] = df_new['current_aqi_value']
    df_new['pm25'] = df_new['current_pm25']
    df_new['pm10'] = df_new['current_pm10']
    df_new['o3'] = df_new['current_o3']
    df_new['co'] = df_new['current_co']
    df_new['so2'] = df_new['current_so2']
    df_new['no2'] = df_new['current_no2']


    # 3. Handle Categorical Features (One-Hot Encoding)
    original_categorical_cols = ['origin', 'destination', 'summary_route']

    temp_df_categorical = pd.DataFrame({
        'origin': [input_data['origin']],
        'destination': [input_data['destination']],
        'summary_route': [input_data['summary_route']]
    })
    
    df_new_encoded = pd.get_dummies(temp_df_categorical, columns=original_categorical_cols, drop_first=True)

    training_ohe_cols = [col for col in feature_cols if col.startswith(tuple([f'{c}_' for c in original_categorical_cols]))]

    for col in training_ohe_cols:
        df_new[col] = 0 # Initialize all OHE columns to 0

    for col in df_new_encoded.columns:
        if col in df_new.columns:
            df_new[col] = df_new_encoded[col].iloc[0]

    # Drop temp columns
    df_new = df_new.drop(columns=[
        'origin', 'destination', 'summary_route', 'timestamp',
        'current_datetime', 'rounded_timestamp', 'current_aqi_value',
        'current_pm25', 'current_pm10', 'current_o3', 'current_co', 'current_so2', 'current_no2'
    ], errors='ignore')

    # Ensure the final DataFrame has exactly the same columns in the same order as feature_cols
    final_df_for_prediction = pd.DataFrame(columns=feature_cols)
    row_data = {}
    for col in feature_cols:
        if col in df_new.columns:
            row_data[col] = df_new[col].iloc[0]
        else:
            row_data[col] = 0
    
    final_df_for_prediction = pd.DataFrame([row_data])

    for col in final_df_for_prediction.columns:
        if final_df_for_prediction[col].dtype == bool:
            final_df_for_prediction[col] = final_df_for_prediction[col].astype(int)

    return final_df_for_prediction

# --- FastAPI Startup Event ---
@app.on_event("startup")
async def startup_event():
    """
    Load the model and feature columns when the FastAPI application starts.
    """
    global model, feature_columns
    print("FastAPI app starting up. Loading model and features...")
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        if not os.path.exists(PROCESSED_DATA_FOR_COLUMNS):
            raise FileNotFoundError(f"Processed data file not found at {PROCESSED_DATA_FOR_COLUMNS}. Needed for feature consistency.")

        model = joblib.load(MODEL_PATH)

        df_processed_ref = pd.read_csv(PROCESSED_DATA_FOR_COLUMNS)
        features_to_exclude_from_raw = [
            'timestamp',
            'travel_time_seconds',
            'distance_meters',
            'speed_kmph', # Excluded during training
            'travel_time_minutes'
        ]
        feature_columns = df_processed_ref.drop(columns=features_to_exclude_from_raw, errors='ignore').columns.tolist()
        
        print("Model and feature columns loaded successfully on startup.")
    except Exception as e:
        print(f"Error loading model or feature columns: {e}")
        # You might want to halt the application or make it return an error on requests
        # For simplicity, we'll let it try to run but log the error.
        raise RuntimeError(f"Failed to load essential resources: {e}") # Raise to prevent startup if critical


# --- Prediction Endpoint ---
@app.post("/predict_travel_time")
async def predict_travel_time_endpoint(request: PredictionRequest):
    """
    Predicts travel time based on provided origin, destination, current conditions, and route summary.
    """
    if model is None or feature_columns is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please check API startup logs.")

    try:
        # Prepare input data for preprocessing
        input_data = request.model_dump() # Converts Pydantic model to a dict
        input_data['current_datetime'] = datetime.now() # Use current time for prediction

        processed_input = preprocess_new_data_for_api(input_data, feature_columns, FETCH_INTERVAL_MINUTES, PROCESSED_DATA_FOR_COLUMNS)

        if processed_input.empty:
            raise HTTPException(status_code=400, detail="Failed to preprocess input data. Check input format.")

        predicted_time = model.predict(processed_input)[0]

        return {"predicted_travel_time_minutes": round(float(predicted_time), 2)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

# --- Root Endpoint (Optional) ---
@app.get("/")
async def read_root():
    return {"message": "Welcome to the Smart Traffic Flow Optimizer API"}