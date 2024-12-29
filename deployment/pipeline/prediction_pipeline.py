from deployment.custom_logging import info_logger, error_logger
from deployment.exception import PredictionError, handle_exception

from deployment.components.feature_engineering import FeatureEngineering
from deployment.components.prediction import Prediction

import os
import sys
from pathlib import Path
import numpy as np


class PredictionPipeline:
    def __init__(self):
        pass

    def predict(self, input):
        try:
            feature_engineering = FeatureEngineering()
            transformed_data = feature_engineering.transform_text(input)

            prediction = Prediction()
            predicted_price = prediction.predict(transformed_data)

            return predicted_price
        except Exception as e:
            handle_exception(e,PredictionError)

if __name__ == "__main__":
    prediction_pipeline = PredictionPipeline()
    input ="K I'm going to bed now. Nite tweets "
    prediction_pipeline = prediction_pipeline.predict(input)
    print(prediction_pipeline)