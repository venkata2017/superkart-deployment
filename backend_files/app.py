from flask import Flask, request, jsonify
import pandas as pd
import joblib

model = joblib.load("superkart_model.joblib")

# Fix: Change variable name from 'app' to 'superkart_api'
superkart_api = Flask(__name__)

FEATURE_COLUMNS = [
    "Product_Weight",
    "Product_Sugar_Content",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
    "Product_Id_char",
    "Store_Age_Years",
    "Product_Type_Category"
]

@superkart_api.get("/")
def home():
    return "<h1>SuperKart Sales prediction API is running</h1>"

@superkart_api.post("/v1/predict")
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json()
        # Fix: Use 'data' instead of undefined variable 'payload'
        input_df = pd.DataFrame([data], columns=FEATURE_COLUMNS)
        prediction = model.predict(input_df)[0]
        return jsonify({"prediction": float(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)})

@superkart_api.post("/v1/predictbatch")
def predictbatch():
    try:
        # Get the file from the request
        file = request.files["file"]
        input_df = pd.read_csv(file)
        predictions = model.predict(input_df)
        output = {str(index): float(prediction) for index, prediction in enumerate(predictions)}
        return jsonify(output)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    superkart_api.run(host="0.0.0.0", port=7860)
