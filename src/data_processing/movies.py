import pandas as pd

from configs.processing_config import DataProcessingConfig
from src.utils.utils import timeit


class ProcessMovies:
    def __init__(
        self, movies_df: pd.DataFrame, processing_config: DataProcessingConfig
    ):
        self.movies_df = movies_df
        self.processing_config = processing_config
        self.output = None

    @timeit
    def process_for_training(self):
        self.movies_df["year"] = self.movies_df["title"].str.extract(r"\((\d{4})\)")
        self.output = self.movies_df[self.processing_config.data_movie_features]
        return self

    @timeit
    def process_for_inference(self):
        self.movies_df["year"] = self.movies_df["title"].str.extract(r"\((\d{4})\)")
        self.output = self.movies_df[self.processing_config.data_movie_features]
        return self
