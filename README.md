# Flask Time Series Prediction API

This is a Flask-based API that provides predictions based on time-series data using a machine learning model (XGBoost). The API allows users to send data, process it with time series feature engineering, and return predictions from a pre-trained model.

## Prerequisites

- **Python Version**: 3.12.7 (Ensure you have this version installed on your system)
- **Flask**: Web framework to serve the API
- **XGBoost**: Machine learning library for building and using models
- **Pandas**: For data manipulation
- **NumPy**: For numerical computations

## Setup Instructions

1. **Clone the Repository**:  
   Clone this repository to your local machine.

   ```bash
     https://github.com/ziadwaelai/GIZA_TASk3.git
   ```

2. **Create a Virtual Environment**:  
   It is recommended to create a virtual environment to manage dependencies.

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install Dependencies**:  
   Use the provided `requirements.txt` file to install all the necessary dependencies.

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Flask App**:  
   After setting up the environment, you can run the Flask API server.

   ```bash
   python server.py
   ```

   The API will start and listen on `http://127.0.0.1:5000/`.

## API Endpoint

- **Endpoint**: `/predict`
- **Method**: `POST`
- **Content-Type**: `application/json`

### Request Body Format

The request body should be a JSON object with the following keys:
- `dataset_id`: The ID of the dataset for which you want to make a prediction.
- `values`: An array of time series data points, with each point containing a `time` (timestamp) and `value` (the feature you are predicting from).
- `anomly`: Some datasets have an additional column **anomly** which is a binary column that indicates if the value is an anomly or not.

Examples JSON request:

```json
{
  "dataset_id": "123",
  "values": [
    {"time": "2021-09-02 00:00:00", "value": 0.4929044474165175},
    {"time": "2021-09-02 06:00:00", "value": null},
    {"time": "2021-09-02 12:00:00", "value": 0.5068339933323015},
    {"time": "2021-09-02 18:00:00", "value": 0.566204629548143}
  ]
}
```
```json
{
  "dataset_id": "9",
  "values": [
      {"time": "2021-09-02 00:00:00", "value": 0.4929044474165175,"anomly":1},
      {"time": "2021-09-02 06:00:00", "value": null,"anomly":0},
      {"time": "2021-09-02 12:00:00", "value": 0.5068339933323015,"anomly":0},
      {"time": "2021-09-02 18:00:00", "value": 0.566204629548143,"anomly":1}
  ]
}
```

### Example Response

The response will include the predicted value based on the input data.

Example response:
```json
{
  "prediction": 0.734554621227935
}
```

### Error Handling

In case of errors, the API will return a JSON response with an error message and a 500 status code.

```json
{
  "error": "Model for Data ID 123 not found."
}
```

## Testing with Postman

1. **Download and Install Postman**:  
   If you haven't already, download and install Postman from [here](https://www.postman.com/downloads/).

2. **Create a New Request**:
   - Open Postman and click **New** -> **HTTP Request**.
   - Set the request method to **POST**.
   - Enter the URL: `http://127.0.0.1:5000/predict`.

3. **Add Headers**:
   - In the **Headers** tab, add the following header:
     - `Content-Type`: `application/json`

4. **Add Body**:
   - Select **Body** -> **raw**.
   - Paste your JSON request in the body as shown in the example above.

5. **Send the Request**:  
   - Click **Send**.
   - You should see the response with the prediction or any error message.

## requirements.txt

Here is the `requirements.txt` file with all the dependencies needed for your project:

```plaintext
Flask
pandas
numpy
xgboost
```

### Additional Notes
- Ensure you have the pre-trained models in the appropriate folder (`Task3/models/`) and a `summary.json` file that contains information on the best lag and window sizes for each dataset ID.
- You can deploy this API to platforms like Heroku or Render for hosting in production.
