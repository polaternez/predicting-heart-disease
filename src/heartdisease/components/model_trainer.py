import os
import sys
from pathlib import Path

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

from heartdisease.entity import ModelTrainerConfig
from heartdisease import logger
from heartdisease.utils.exception import CustomException
from heartdisease.utils.ml_helper import save_object, evaluate_models


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.model_trainer_config = config
    
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logger.info("Split data into training and testing sets")
            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = (
                train_array[:, :-1],
                test_array[:, :-1],
                train_array[:, -1],
                test_array[:, -1]
            )

            # Define models and hyperparameters 
            models_dict = {
                "XGBClassifier": XGBClassifier(random_state=42),
                "CatBoost Classifier": CatBoostClassifier(verbose=False, random_state=42),
                "Logistic Regression": LogisticRegression(),
                "Nearest Neighbors": KNeighborsClassifier(),
                "Naive Bayes": GaussianNB(),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "Random Forest": RandomForestClassifier(random_state=42),
                "AdaBoost Classifier": AdaBoostClassifier(random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42),
            }
            params_dict = {
                "XGBClassifier": {
                    'learning_rate': [0.01, 0.03, 0.1, 0.3],
                    'n_estimators': [50, 100, 150]
                },
                "CatBoost Classifier": {},
                "Logistic Regression": {},
                "Nearest Neighbors": {},
                "Naive Bayes": {},
                "Decision Tree": {},
                "Random Forest": {},
                "AdaBoost Classifier": {},
                "Gradient Boosting": {},
            }

            logger.info(f"Starting model performance assessment...")
            # Evaluate all models
            model_report = evaluate_models(models_dict=models_dict, params_dict=params_dict,
                                           X=X_train, y=y_train,
                                           validation_data=(X_test, y_test))
            
            for model, score in model_report.items():
                print("[{}] - Accuracy: {:.4f}".format(model, score))
            
            ## To get best model name from dict
            best_model_name = sorted(model_report, key=lambda k: model_report[k], reverse=True)[0]
            best_model_score = model_report[best_model_name]

            # the best model
            best_model = models_dict[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No model achieved satisfactory performance.")
            
            logger.info(f"The best model found on both the training and testing dataset")

            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_path,
                obj=best_model
            )
            logger.info("Model saved")

            predicted = best_model.predict(X_test)
            acc_score = accuracy_score(y_test, predicted)
            return acc_score
        except Exception as e:
            raise CustomException(e, sys)