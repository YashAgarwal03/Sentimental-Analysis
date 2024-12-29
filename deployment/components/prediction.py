from deployment.custom_logging import info_logger, error_logger
from deployment.exception import PredictionError, handle_exception
import numpy as np
import joblib
import os
import sys
from pathlib import Path

class Prediction:
    def __init__(self):
        pass

    def predict(self, data):
        try:
            model_path = "artifacts/model_trainer/final_model.joblib"
            model = joblib.load(model_path)

            predicted_price = model.predict(data)

            return predicted_price
        except Exception as e:
            handle_exception(e, PredictionError)

