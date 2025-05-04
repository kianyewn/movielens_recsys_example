import pandas as pd

from configs.data_config import InferenceDataConfig
from configs.processing_config import DataProcessingConfig
from src.feature_processing.process_features import ProcessFeatures

if __name__ == "__main__":
    # load config
    data_config = InferenceDataConfig()
    processing_config = DataProcessingConfig()

    # load data
    users = pd.read_parquet(data_config.processed_users.file_path)
    movies = pd.read_parquet(data_config.processed_movies.file_path)
    ratings = pd.read_parquet(data_config.processed_ratings.file_path)

    # process data
    pf_object = ProcessFeatures(users, movies, ratings, processing_config)
    pf_object_loaded = pf_object.load()
    pf_object_loaded.process_for_inference()
    pf_object_loaded.inference_data.shape

    # save data
    pf_object_loaded.inference_data.to_parquet(
        data_config.inference_model_input.file_path
    )
