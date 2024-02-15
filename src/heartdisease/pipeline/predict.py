import os
import sys
from pathlib import Path
import pandas as pd

from heartdisease.utils.ml_helper import load_object
from heartdisease.utils.exception import CustomException


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            preprocessor_path = Path('artifacts/preprocessing/preprocessor.pkl')
            model_path = Path("artifacts/models/model.pkl")

            print("Before Loading")
            preprocessor = load_object(file_path=preprocessor_path)
            model = load_object(file_path=model_path)
            print("After Loading")
            
            data_transformed = preprocessor.transform(features)
            prediction = model.predict(data_transformed)
            return prediction
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self, Age: int, Sex: str, ChestPainType: str, RestingBP: int,
                 Cholesterol: int, FastingBS: int, RestingECG: str, MaxHR: int,
                 ExerciseAngina: str, Oldpeak: int, ST_Slope: str):
        self.Age = Age
        self.Sex = Sex
        self.ChestPainType = ChestPainType
        self.RestingBP = RestingBP
        self.Cholesterol = Cholesterol
        self.FastingBS = FastingBS
        self.RestingECG = RestingECG
        self.MaxHR = MaxHR
        self.ExerciseAngina = ExerciseAngina
        self.Oldpeak = Oldpeak
        self.ST_Slope = ST_Slope

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "Age": [self.Age],
                "Sex": [self.Sex],
                "ChestPainType": [self.ChestPainType], 
                "RestingBP": [self.RestingBP], 
                "Cholesterol": [self.Cholesterol], 
                "FastingBS": [self.FastingBS], 
                "RestingECG": [self.RestingECG],
                "MaxHR": [self.MaxHR], 
                "ExerciseAngina": [self.ExerciseAngina],
                "Oldpeak": [self.Oldpeak], 
                "ST_Slope": [self.ST_Slope]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)