import os 
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json


from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold, train_test_split

from training.custom_logging import info_logger,error_logger
from training.exception import CrossValError, handle_exception

from training.configuration_manager.configuration import ConfigurationManager
from training.entity.config_entity import CrossValConfig

class CrossVal:

    def __init__(self,config: CrossValConfig):
        self.config = config

    @staticmethod
    def is_json_serializable(value):
        """
        Check if a vale is JSON serializable.
        """
        try:
            json.dumps(value)
            return True
        except (TypeError,OverflowError):
            return False
    
    def load_ingested_data(self):
        try:
            info_logger.info("Cross Validation Component Started")
            info_logger.info("Loading ingested Data")

            transform_data_path = os.path.join(self.config.transform_data, 'transform.npy')
            target_column_path = os.path.join(self.config.target,"traget.npy")

            X = np.load(transform_data_path, allow_pickle=True)

            y = np.load(target_column_path, allow_pickle=True)
                

            info_logger.info('Ingested data loaded')
            return X, y
        
        except Exception as e:
            handle_exception(e,CrossValError)

    def split_data_for_final_train(self, X, y):
        try:
            info_logger.info("Data Split for final train started")

            xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

            info_logger.info("Data split for final train completed")
            return xtrain, xtest, ytrain, ytest
        
        except Exception as e:
            handle_exception(e,CrossValError)
        

    def save_data_for_final_train(self,xtrain, xtest, ytrain, ytest ):
        try:
            info_logger.info('Saving Data for final train started')

            final_train_data_path = self.config.final_train_data_path
            final_test_data_path = self.config.final_test_data_path

            # save X_train and y_train to train.npz
            # sace X_test and y_test to test.npz
            np.savez(os.path.join(final_train_data_path,'Train.npz'), xtrain=xtrain,ytrain=ytrain)
            np.savez(os.path.join(final_test_data_path,'Test.npz'), xtest=xtest,ytest=ytest)
            info_logger.info('Saving Data for final train complete')
        except Exception as e:
            handle_exception(e, CrossValError)

    def run_cross_val(self,X, y):
        try:
            info_logger.info("Cross Validation Sarted")

            # Step 4: Create a pipeline with Linear Regression
            pipeline = Pipeline(steps=[
                ('classifier', LogisticRegression())
                ])

            # Set up GridSearchCV for hyperparameter tuning
            param_grid = {
                'classifier__C': [0.1, 1.0, 10.0],       # Regularization strength
                'classifier__penalty': ['l2'],           # Regularization type
                'classifier__solver': ['lbfgs', 'liblinear']  # Solvers
                }

            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                scoring='accuracy',  # Use accuracy for evaluation
                cv=5,
                verbose=2
                )

            # Step 6: Fit GridSearchCV on the training data
            grid_search.fit(X, y)

            best_model = grid_search.best_estimator_
            best_params =grid_search.best_params_
            best_scores = grid_search.best_score_

            with open(self.config.STATUS_FILE,"a") as f:
                f.write(f"Best params for Model: {str(best_params)}\n")
                f.write(f"Best scoring(accuracy) for Model: {str(best_scores)}")

            best_model_params_path = os.path.join(self.config.best_model_params,f"best_params.json")
            best_model_params = best_model.get_params()
            serializable_params = {k: v for k, v in best_model_params.items() if self.is_json_serializable(v)}

            with open(best_model_params_path,'w') as f:
                json.dump(serializable_params, f,indent=4)


            info_logger.info("Cross Validation complete")
        except Exception as e:
            handle_exception(e, CrossValError)



if __name__ == '__main__':
    config = ConfigurationManager()
    cross_val_config = config.get_cross_val_config()

    cross_val = CrossVal(config=cross_val_config)

    #Load the feature and Target
    X,y = cross_val.load_ingested_data()

    #Split the data into train and test sets for final train
    xtrain, xtest, ytrain, ytest = cross_val.split_data_for_final_train(X,y)

    #Save xtrain, xtest, ytrain, ytest to be used final train
    cross_val.save_data_for_final_train(xtrain,xtest,ytrain,ytest)

    # Run cross validation
    cross_val.run_cross_val(xtrain, ytrain)
