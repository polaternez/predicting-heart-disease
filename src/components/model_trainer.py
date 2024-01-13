import os
import sys

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "XGBClassifier": XGBClassifier(),
                "CatBoost Classifier": CatBoostClassifier(verbose=False),
                "Logistic Regression": LogisticRegression(),
                "Nearest Neighbors": KNeighborsClassifier(),
                "Naive Bayes": GaussianNB(),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "AdaBoost Classifier": AdaBoostClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
            }

            params = {
                "XGBClassifier": {},
                "CatBoost Classifier": {},
                "Logistic Regression": {},
                "Nearest Neighbors": {},
                "Naive Bayes": {},
                "Decision Tree": {},
                "Random Forest": {},
                "AdaBoost Classifier": {},
                "Gradient Boosting": {},
            }

            model_report = evaluate_models(X=X_train, y=y_train,
                                           validation_data=(X_test, y_test),
                                           models_dict=models, params_dict=params)
            print(model_report)
            
            ## To get best model name from dict
            best_model_name = sorted(model_report, key=lambda k: model_report[k], reverse=True)[0]
            best_model_score = model_report[best_model_name]

            # the best model
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            
            logging.info(f"Best model found on both the training and testing dataset")

            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            acc_score = accuracy_score(y_test, predicted)
            
            return acc_score
            
        except Exception as e:
            raise CustomException(e, sys)