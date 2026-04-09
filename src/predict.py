# src/predict.py

import numpy as np
import joblib

def load_artifacts(model_path, scaler_path):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def predict(model, scaler, input_data):
    input_data = np.array(input_data).reshape(1, -1)

    # Scale Time & Amount
    input_data[:, [0, 1]] = scaler.transform(input_data[:, [0, 1]])

    prob = model.predict_proba(input_data)[0][1]
    pred = model.predict(input_data)[0]

    return pred, prob