import os
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from heartdisease.entity import DataIngestionConfig
from heartdisease import logger
from heartdisease.utils.exception import CustomException


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.ingestion_config = config

    def initiate_data_ingestion(self):
        logger.info("Initiating data ingestion")

        try:
            dataset = pd.read_csv(self.ingestion_config.source_file)
            logger.info('Dataset loaded')

            # Create a directory
            self.ingestion_config.root_dir.mkdir(parents=True, exist_ok=True)
            dataset.to_csv(self.ingestion_config.raw_data_path, index=False)
            logger.info("Dataset saved")

            train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
            train_data.to_csv(self.ingestion_config.train_data_path, index=False)
            test_data.to_csv(self.ingestion_config.test_data_path, index=False)
            logger.info("Data ingestion completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)
        
    
        


    
