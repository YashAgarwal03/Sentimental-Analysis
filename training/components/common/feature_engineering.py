import os
import sys
import pandas as pd
import numpy as np

from training.exception import FeatureEngineeringError,handle_exception
from training.custom_logging import info_logger, error_logger

from training.entity.config_entity import FeatureEngineeringConfig
from training.configuration_manager.configuration import ConfigurationManager
from nltk.corpus import stopwords
import re 

class FeatureEngineering:
    def __init__(self, config:FeatureEngineeringConfig ):
        self.config = config
        self.stop_words = set(stopwords.words('english'))
        self.df = None
        
    def loading_data(self):
        try:

            info_logger.info("Loading data for clean the text")
                
            data_path = self.config.source

            self.df = pd.read_csv(data_path, index_col = 0)
                
            info_logger.info("Data loaded")

        except Exception as e:
            handle_exception(e, FeatureEngineeringError)

    def clean_text(self, text):
        text = re.sub(r'http\S+', '', text)  # Remove URLs
        text = re.sub(r'@\w+', '', text)     # Remove mentions
        text = re.sub(r'#\w+', '', text)     # Remove hashtags
        text = re.sub(r'\d+', '', text)      # Remove numbers
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = text.lower()                  # Convert to lowercase
        text = text.split()
        text = [word for word in text if word not in self.stop_words]
        return ' '.join(text)

    def apply_transformation(self):
        try:
            info_logger.info("clean the text column")
            self.df['clean_text'] = self.df['text'].apply(self.clean_text)

            info_logger.info("cleaning complete ")

        except Exception as e:
            handle_exception(e, FeatureEngineeringError)
    def save_data(self):
        try:
            info_logger.info("save the new data my new column name clean_text as preprocessed_data.csv")

            self.df = self.df[["target","clean_text"]]
            
            data_file_path = os.path.join(self.config.data_dir,"preprocessed_data.csv")
            self.df.to_csv(data_file_path, index=False)

            info_logger.info("data saved")
            
            with open(self.config.STATUS_FILE,"w") as f:
                f.write(f"Feature Engineering status: True")
        
        except Exception as e:
            handle_exception(e, FeatureEngineeringError)

if __name__ == "__main__":
    config = ConfigurationManager()
    feature_engineering_config = config.get_feature_engineering_config()

    feature_engineering = FeatureEngineering(config = feature_engineering_config)
    feature_engineering.loading_data()
    feature_engineering.apply_transformation()
    feature_engineering.save_data()