

import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model and scaler
regmodel = pickle.load(open("regmodel.pkl", "rb"))
scalar = pickle.load(open("sacling.pkl", "rb"))

# Use the same column names used during training
feature_names = ['crim', 'zn', 'indus', 'chas', 'nox', 'rm',
                 'age', 'dis', 'rad', 'tax', 'ptratio', 'black', 'lstat']  # Replace with your actual feature names

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)

    # Convert to DataFrame with proper feature names
    new_data_df = pd.DataFrame([data], columns=feature_names)
    new_data_scaled = scalar.transform(new_data_df)
    
    output = regmodel.predict(new_data_scaled)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict', methods=["POST"])
def predict():
    data = [float(x) for x in request.form.values()]
    new_data_df = pd.DataFrame([data], columns=feature_names)
    final_input = scalar.transform(new_data_df)

    print(final_input)
    output = regmodel.predict(final_input)
    return render_template("home.html", prediction_txt =f"The House Price Prediction is {output[0]:.2f}")

if __name__ == "__main__":
    app.run(debug=True)
    
    
    
    
    