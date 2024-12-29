import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

from training.exception import FeatureExtractionError,handle_exception
from training.custom_logging import info_logger, error_logger

from training.entity.config_entity import FeatureExtractionConfig
from training.configuration_manager.configuration import ConfigurationManager


class FeatureExtraction:
    def __init__(self, config:FeatureExtractionConfig ):
        self.config = config
        self.tfidf =  TfidfVectorizer(max_features=5000)
        self.df = None
        self.X = None
        self.y = None

    def loading_data(self):
        try:
            info_logger.info("Loading data for clean the text")
            
            data_path = self.config.source
            self.df = pd.read_csv(data_path)
            
            info_logger.info("Data loaded")
        except Exception as e:
            handle_exception(e, FeatureExtractionError)

    def tfidf_vectorizer(self):
        try:
            info_logger.info("tfidf vectorizer starts")
            self.df['clean_text'] = self.df['clean_text'].fillna('')
            
            self.X = self.tfidf.fit_transform(self.df['clean_text']).toarray()

            self.y = self.df["target"].values

            vectorizer_path = os.path.join(self.config.root_dir, "tfidf_vectorizer.joblib")
            joblib.dump(self.tfidf, vectorizer_path)

            info_logger.info("tfidf vectorizer complete")

        except Exception as e:
            handle_exception(e, FeatureExtractionError)

    def save_data(self):
        try:
            info_logger.info("saving the tranform data and target numpy array")
            
            data_file_path = os.path.join(self.config.transform_data,"transform.npy")
            np.save(data_file_path, self.X)

            target_file_path = os.path.join(self.config.target,"traget.npy")
            np.save(target_file_path, self.y)
            
            info_logger.info("data saved")

            with open(self.config.STATUS_FILE,"w") as f:
                f.write(f"Feature Extraction status: True")

        except Exception as e:
            handle_exception(e, FeatureExtractionError)
            

if __name__ == "__main__":
    config = ConfigurationManager()
    feature_extraction_config = config.get_feature_extraction_config()

    feature_extraction = FeatureExtraction(config = feature_extraction_config)
    feature_extraction .loading_data()
    feature_extraction .tfidf_vectorizer()
    feature_extraction .save_data()