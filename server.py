from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import xgboost as xgb
import os
import json

# Initialize Flask app
app = Flask(__name__)
# Load your model (example function, replace with your actual model loading logic)
def load_model(data_id):
    model_path = f'Task3/models/model_{data_id}.json'
    model = xgb.XGBRegressor()
    print(model_path)
    if os.path.exists(model_path):
        model.load_model(model_path)
        print(f"Model for Data ID {data_id} loaded successfully.")
        return model
    else:
        return None

# handle null values
def handle_null_values(df):
    df.ffill(inplace=True)  # Forward fill
    # df.bfill( inplace=True)  # Backward fill
    
    return df

def time_features(df):
    """
    Create time series features based on time series index.
    """
    df = df.set_index('timestamp')
    df.index = pd.to_datetime(df.index)
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week

    return df

# lag features
def lag_features(df, lags=7):
    for lag in range(1, lags + 1):
        df[f'lag({lag})'] = df['value'].shift(lag)
    return df

# rolling window features
def rolling_window_features(df, windows=7):
    for window in range(2, windows + 1):
        df[f'rolling_mean({window})'] = df['value'].rolling(window=window).mean()
        df[f'rolling_std({window})'] = df['value'].rolling(window=window).std()
    return df

def pipeline(df, lags=7, windows=7,train=False):
    # shift the value must the target for the next time step
    df['value'] = df['value'].shift(-1)
    df.dropna(axis=1, how='all', inplace=True) # Drop columns with all NaN values
    df = handle_null_values(df)
    df = time_features(df)
    best_lag, best_window = lags, windows
    df = lag_features(df, lags=int(best_lag))
    df = rolling_window_features(df, windows=int(best_window))
    if 'anomaly' in df.columns:
        df['anomaly'] = df['anomaly'].astype(int)
    df = handle_null_values(df)
    df.dropna(inplace=True)
    return df

# Prediction function
def make_prediction(request_body):
    # Extract dataset_id and values from the request body
    dataset_id = request_body['dataset_id']
    values = request_body['values']
    
    # Load the corresponding model
    model = load_model(dataset_id)
    if model is None:
        raise ValueError(f"Model for Data ID {dataset_id} not found.")
    
    # Convert the input values into a DataFrame
    df_input = pd.DataFrame(values)
    df_input = df_input.rename(columns={'time': 'timestamp'})  # Rename for consistency

    # load the summary file
    with open('Task3/summary.json', 'r') as f:
        summary = json.load(f)

    # Find the best lag and window for the dataset
    best_lag = None
    best_window = None
    for data in summary:
        if data['data_id'] == dataset_id:
            best_lag = data['best_lag']
            best_window = data['best_window']
            break
    
    # Assuming you have a preprocessing function (pipeline) defined
    df_input= pipeline(df_input,best_lag,best_window,train=False)  # Apply feature extraction or preprocessing

    df_input = df_input.drop(['value'], axis=1)  # Drop the 'value' column (if present)

    # Ensure the input DataFrame has the expected features
    expected_features = model.get_booster().feature_names
    missing_features = set(expected_features) - set(df_input.columns)
    if missing_features:
        raise ValueError(f"Missing expected features: {missing_features}")
    
    # Make prediction
    prediction = model.predict(df_input)

    # Prepare the response
    if len(prediction) ==0:
        response = {'prediction': 'no prediction,must have at least {} previous data'.format(max(best_lag,best_window)+1)}
    else:
        response = {'prediction': prediction.tolist()[-1]} 
    
    return response

# Define an endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        request_body = request.get_json()

        # Call the make_prediction function
        response = make_prediction(request_body)
        
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    print("Starting Flask API server...")
    app.run(port=5000, debug=False)