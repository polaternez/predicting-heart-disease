import os
import sys
import numpy as np 
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import dill
import pickle

from src.exception import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)

    
def evaluate_models(X, y, validation_data: tuple, models_dict: dict, params_dict: dict) -> dict:
    try:
        report = {}

        for model_name, model in models_dict.items():
            params = params_dict[model_name]

            grid_search = GridSearchCV(model, params, cv=3)
            grid_search.fit(X, y)

            # best model
            model.set_params(**grid_search.best_params_)
            model.fit(X, y)

            # predictions
            y_train_pred = model.predict(X)
            y_test_pred = model.predict(validation_data[0])

            # evaluation results
            train_model_score = accuracy_score(y, y_train_pred)
            test_model_score = accuracy_score(validation_data[1], y_test_pred)

            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)