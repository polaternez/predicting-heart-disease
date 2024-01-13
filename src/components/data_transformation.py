import os
import sys
import numpy as np 
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        '''
        try:
            num_cols = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
            cat_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

            # build pipeline
            num_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            cat_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(drop="if_binary"))
            ])

            logging.info(f"Categorical columns: {cat_cols}")
            logging.info(f"Numerical columns: {num_cols}")

            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, num_cols),
                ("cat_pipelines", cat_pipeline, cat_cols)
            ])

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
    
     
    def initiate_data_transformation(self, train_path, test_path):
        try:
            target_column_name = "HeartDisease"
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            target_feature_train_df = train_df[target_column_name]
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            
            target_feature_test_df = test_df[target_column_name]
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)

            # the preprocesing object created
            preprocessing_obj = self.get_data_transformer_object()

            logging.info("The preprocesing object created")

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Save the preprocessing object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info(f"Preprocessing object saved.")
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        
        except Exception as e:
            raise CustomException(e, sys)

