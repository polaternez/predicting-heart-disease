import os
import sys
from pathlib import Path
import numpy as np 
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from dataclasses import dataclass
from src.utils.exception import CustomException
from src.utils.logger import logging
from src.utils.ml_helper import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_file_path: Path = Path('artifacts/preprocessing/preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_preprocessor(self):
        '''
        Creates a data preprocessor
        '''
        num_cols = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
        cat_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

        logging.info(f"Categorical columns: {cat_cols}")
        logging.info(f"Numerical columns: {num_cols}")

        # Define numerical pipeline
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        # Define categorical pipeline
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(drop="if_binary"))
        ])

        preprocessor = ColumnTransformer([
            ("num_pipeline", num_pipeline, num_cols),
            ("cat_pipelines", cat_pipeline, cat_cols)
        ])
        return preprocessor
    
    def initiate_data_transformation(self, train_path, test_path):
        """
        Transforms data and saves the preprocessor
        """
        try:
            target_column = "HeartDisease"
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Train/test data reading complete")

            y_train = train_df[target_column]
            X_train = train_df.drop(columns=[target_column], axis=1)
            y_test = test_df[target_column]
            X_test = test_df.drop(columns=[target_column], axis=1)

            # Create the data preprocessor
            preprocessor = self.get_preprocessor()
            logging.info("Preprocessor is created")

            logging.info("Applying the preprocessor to the training dataframe and testing dataframe")
            X_train_processed = preprocessor.fit_transform(X_train)
            X_test_processed = preprocessor.transform(X_test)

            # Combine features and target
            train_arr = np.c_[X_train_processed, np.array(y_train)]
            test_arr = np.c_[X_test_processed, np.array(y_test)]

            # Save the preprocessor
            save_object(
                file_path=self.data_transformation_config.preprocessor_file_path,
                obj=preprocessor
            )
            logging.info(f"Preprocessor saved.")
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)

