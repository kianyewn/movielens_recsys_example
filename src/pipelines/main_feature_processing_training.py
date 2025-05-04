from configs.data_config import DataConfig
from configs.processing_config import DataProcessingConfig
import pandas as pd
from src.feature_processing.process_features import ProcessFeatures


if __name__ == "__main__":
    # load config
    data_config = DataConfig()
    processing_config = DataProcessingConfig()

    # load data
    users = pd.read_parquet(data_config.processed_users.file_path)
    movies = pd.read_parquet(data_config.processed_movies.file_path)
    ratings = pd.read_parquet(data_config.processed_ratings.file_path)

    # feature process data
    pf_object = ProcessFeatures(users, movies, ratings, processing_config)
    pf_object.process_for_training()

    # save processed data
    pf_object.train_data.to_parquet(data_config.train.file_path)
    pf_object.val_data.to_parquet(data_config.val.file_path)
    pf_object.test_data.to_parquet(data_config.test.file_path)
