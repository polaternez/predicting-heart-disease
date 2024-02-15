from heartdisease.config.configuration import ConfigurationManager
from heartdisease.components.data_ingestion import DataIngestion
from heartdisease.components.data_transformation import DataTransformation
from heartdisease.components.model_trainer import ModelTrainer


def main():
    config_manager = ConfigurationManager()
    data_ingestion_config = config_manager.get_data_ingestion_config()
    data_ingestion = DataIngestion(data_ingestion_config)
    train_path, test_path = data_ingestion.initiate_data_ingestion()

    data_transformation_config = config_manager.get_data_transformation_config()
    data_transformation = DataTransformation(data_transformation_config)
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_path, test_path)
    
    model_trainer_config = config_manager.get_model_trainer_config()
    model_trainer = ModelTrainer(model_trainer_config)
    print(model_trainer.initiate_model_trainer(train_arr, test_arr))


if __name__ == "__main__":
    main()