from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import xgboost as xgb
import os
import json

# Initialize Flask app
app = Flask(__name__)
# Load your model
def load_model(data_id):
    model_path = f'models/model_{data_id}.json'
    model = xgb.XGBRegressor()
    print(model_path)
    if os.path.exists(model_path):
        model.load_model(model_path)
        print(f"Model for Data ID {data_id} loaded successfully.")
        return model
    else:
        print(f"yesss")
        return None

# handle null values
def handle_null_values(df):
    df.ffill(inplace=True)  # Forward fill
    # df.bfill( inplace=True)  # Backward fill
    
    return df

def time_features(df):

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
    df = handle_null_values(df)
    # shift the value must the target for the next time step
    df['value'] = df['value'].shift(-1)
    df.dropna(axis=1, how='all', inplace=True) # Drop columns with all NaN values
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
    dataset_id = request_body['dataset_id']
    values = request_body['values']
    
    model = load_model(dataset_id)
    if model is None:
        raise ValueError(f"Model for Data ID {dataset_id} not found.")
    
    df_input = pd.DataFrame(values)
    df_input = df_input.rename(columns={'time': 'timestamp'})  # Rename for consistency

    with open('summary.json', 'r') as f:
        summary = json.load(f)

    best_lag = None
    best_window = None
    for data in summary:
        if data['data_id'] == dataset_id:
            best_lag = data['best_lag']
            best_window = data['best_window']
            break
    
    df_input= pipeline(df_input,best_lag,best_window,train=False)  

    df_input = df_input.drop(['value'], axis=1) 

    expected_features = model.get_booster().feature_names
    missing_features = set(expected_features) - set(df_input.columns)
    if missing_features:
        raise ValueError(f"Missing expected features: {missing_features}")
    
    prediction = model.predict(df_input)

    if len(prediction) ==0:
        raise ValueError( 'no prediction,must have at least {} previous data'.format(max(best_lag,best_window)+1))
    else:
        response = {'prediction': prediction.tolist()[-1]} 
    
    return response

# Define an endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        request_body = request.get_json()

        response = make_prediction(request_body)
        
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    print("Starting Flask API server...")
    app.run(port=5000, debug=False)