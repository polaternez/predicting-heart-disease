import os
import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from dataclasses import dataclass
from src.utils.exception import CustomException
from src.utils.logger import logging
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig


@dataclass
class DataIngestionConfig:
    data_dir: Path = Path('artifacts/data')
    raw_data_path: Path = data_dir / 'data.csv'
    train_data_path: Path = data_dir / 'train.csv'
    test_data_path: Path = data_dir / 'test.csv'
    

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Initiating data ingestion")

        try:
            dataset = pd.read_csv('notebooks\data\heart.csv')
            logging.info('Dataset loaded')

            # Create a directory
            self.ingestion_config.data_dir.mkdir(parents=True, exist_ok=True)
            dataset.to_csv(self.ingestion_config.raw_data_path, index=False)

            logging.info("Train test split initiated")
            train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
            train_data.to_csv(self.ingestion_config.train_data_path, index=False)
            test_data.to_csv(self.ingestion_config.test_data_path, index=False)
            logging.info("Data ingestion completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)
        


    
