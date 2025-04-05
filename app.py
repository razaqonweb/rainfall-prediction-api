import os

from flask import Flask, request
from flask_cors import CORS
import joblib
import pandas as pd
import warnings
from dotenv import load_dotenv

load_dotenv()

warnings.simplefilter("ignore", category=UserWarning)


app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY")

# Enable CORS
CORS(app)

# Load the trained model and feature names from the pickle file
model_data = joblib.load("rainfall_prediction_model.pkl")


# Routes
@app.post("/predict")
def predict():
    model, feature_names = model_data.values()

    # validate inputs (only numbers allowed)
    valid_inputs = []

    try:
        print(request.get_json())
        for value in request.get_json().values():
            value = float(value)
            if value <= 0:
                return {"error": "Input must be greater than zero."}, 400

            valid_inputs.append(value)
    except (ValueError, TypeError) as e:
        print(f"Error: {str(e)}")
        return {"error": "Invalid input. Please provide valid numbers."}, 400

    input_df = pd.DataFrame([valid_inputs], columns=feature_names)

    prediction = model.predict(input_df)
    prediction_result = "Rainfall" if prediction[0] == 1 else "No rainfall"

    return {"prediction": prediction_result}, 200


if __name__ == "__main__":
    app.run(debug=True, host="localhost", port=8080)
