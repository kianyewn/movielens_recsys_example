import pandas as pd

from configs.data_config import InferenceDataConfig
from configs.processing_config import DataProcessingConfig
from src.data_processing.movies import ProcessMovies
from src.data_processing.ratings import ProcessRatings
from src.data_processing.users import ProcessUsers

if __name__ == "__main__":
    # load config
    data_config = InferenceDataConfig()
    processing_config = DataProcessingConfig()

    # load data
    users = pd.read_csv(
        data_config.raw_users.file_path, **data_config.raw_users.read_args
    )
    movies = pd.read_csv(
        data_config.raw_movies.file_path, **data_config.raw_movies.read_args
    )
    ratings = pd.read_csv(
        data_config.raw_ratings.file_path, **data_config.raw_ratings.read_args
    )

    # process data
    process_users = ProcessUsers(users, processing_config)
    process_movies = ProcessMovies(movies, processing_config)
    process_ratings = ProcessRatings(ratings, processing_config)

    process_users.process_for_inference()
    process_movies.process_for_inference()
    process_ratings.process_for_inference()

    # save data
    process_users.output.to_parquet(data_config.processed_users.file_path)
    process_movies.output.to_parquet(data_config.processed_movies.file_path)
    process_ratings.output.to_parquet(data_config.processed_ratings.file_path)
