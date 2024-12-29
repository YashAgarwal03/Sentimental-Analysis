from deployment.custom_logging import info_logger, error_logger
from deployment.exception import FeatureEngineeringError, handle_exception
import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path
import joblib
import re
from nltk.corpus import stopwords

class FeatureEngineering:
    def __init__(self):
        pass

    def transform_text(self, text):
        try:
            # Clean the text
            text = re.sub(r'http\S+', '', text)  # Remove URLs
            text = re.sub(r'@\w+', '', text)     # Remove mentions
            text = re.sub(r'#\w+', '', text)     # Remove hashtags
            text = re.sub(r'\d+', '', text)      # Remove numbers
            text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
            text = text.lower()                  # Convert to lowercase
            text = text.split()
            text = [word for word in text if word not in stopwords.words('english')]
            clean_text = ' '.join(text)

            transformation = "artifacts/feature_extraction/tfidf_vectorizer.joblib"
            transformation = joblib.load(transformation)

            transformed_text = transformation.transform([clean_text]).toarray()

            return transformed_text
        except Exception as e:
            handle_exception(e, FeatureEngineeringError)


if __name__ == "__main__":

    text = "Sun &amp; rain on my way to Frankfurt Airport. Computer starts very slowly. My tie isn't the ideal one for today..."
    feature_engineering = FeatureEngineering()
    transformed_text = feature_engineering.transform_text(text)
    print(transformed_text)