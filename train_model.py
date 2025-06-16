import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV # Import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

# --- Configuration ---
PROCESSED_DATA_INPUT_FILE = "processed_combined_data_hyd.csv"
MODEL_OUTPUT_FILE = "traffic_prediction_model.pkl" # This file will now store the tuned model

def train_traffic_model_tuned(): # Renamed the function to reflect tuning
    """
    Loads processed data, trains an XGBoost model with hyperparameter tuning,
    evaluates it, and saves the best model.
    """
    print("Starting model training process with Hyperparameter Tuning...")

    # 1. Load the processed data
    if not os.path.exists(PROCESSED_DATA_INPUT_FILE):
        print(f"Error: Processed data file '{PROCESSED_DATA_INPUT_FILE}' not found. Please run preprocessing first.")
        return
    
    df = pd.read_csv(PROCESSED_DATA_INPUT_FILE)
    print(f"Loaded {len(df)} processed data points.")

    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # 2. Define Features (X) and Target (y)
    features_to_exclude = [
        'timestamp',
        'travel_time_seconds',
        'distance_meters',
        'speed_kmph',
        'travel_time_minutes'
    ]

    X = df.drop(columns=features_to_exclude, errors='ignore')
    y = df['travel_time_minutes']

    X = X.fillna(0) # Ensure no NaNs in features
    
    print(f"Features (X) shape: {X.shape}")
    print(f"Target (y) shape: {y.shape}")
    print(f"Feature columns: {X.columns.tolist()}")

    # 3. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")

    # 4. Model Training with GridSearchCV for Hyperparameter Tuning
    print("Starting Hyperparameter Tuning with GridSearchCV...")

    # Define the XGBoost Regressor base model
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)

    # Define the parameter grid to search
    # For a quick initial run, keep the grid small.
    # For more thorough tuning, expand the ranges and add more parameters.
    param_grid = {
        'n_estimators': [100, 200], # Number of boosting rounds
        'learning_rate': [0.05, 0.1], # Step size shrinkage
        'max_depth': [3, 5], # Maximum depth of a tree
        # 'subsample': [0.8, 1.0], # Subsample ratio of the training instance
        # 'colsample_bytree': [0.8, 1.0] # Subsample ratio of columns when constructing each tree
    }

    # Initialize GridSearchCV
    # cv: Number of folds for cross-validation (e.g., 5-fold cross-validation)
    # scoring: Metric to optimize (e.g., 'neg_mean_absolute_error' or 'neg_root_mean_squared_error')
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=3, # Using 3-fold cross-validation for speed. Use 5 or 10 for more robustness.
        scoring='neg_mean_absolute_error', # Optimize for MAE (negative for GridSearchCV)
        verbose=1, # Prints progress messages
        n_jobs=-1 # Use all available cores
    )

    # Fit GridSearchCV to the training data
    grid_search.fit(X_train, y_train)

    print("\nHyperparameter Tuning complete.")
    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best MAE (from cross-validation): {-grid_search.best_score_:.2f} minutes") # MAE is neg, so negate

    # Get the best model found by GridSearchCV
    best_model = grid_search.best_estimator_
    print("Best model selected.")

    # 5. Model Evaluation (using the best model on the test set)
    print("\nEvaluating the best model performance on the test set...")
    y_pred = best_model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n--- Best Model Evaluation Results on Test Set ---")
    print(f"Mean Absolute Error (MAE): {mae:.2f} minutes")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f} minutes")
    print(f"R-squared (R2 Score): {r2:.2f}")

    # 6. Model Saving
    print(f"\nSaving tuned model to {MODEL_OUTPUT_FILE}...")
    joblib.dump(best_model, MODEL_OUTPUT_FILE)
    print("Tuned model saved successfully.")

if __name__ == "__main__":
    train_traffic_model_tuned()