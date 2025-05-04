import pandas as pd

from configs.processing_config import DataProcessingConfig
from src.utils.utils import timeit


class ProcessUsers:
    def __init__(self, users_df: pd.DataFrame, processing_config: DataProcessingConfig):
        self.users_df = users_df
        self.output = None
        self.processing_config = processing_config

    @timeit
    def process_for_training(self):
        self.output = self.users_df[self.processing_config.data_user_features]
        return self

    @timeit
    def process_for_inference(self):
        self.output = self.users_df[self.processing_config.data_user_features]
        return self
