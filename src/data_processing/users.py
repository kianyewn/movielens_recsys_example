import pandas as pd

from configs.model_config import ModelConfig
from src.utils.utils import timeit


class ProcessUsers:
    def __init__(self, users_df: pd.DataFrame, model_config: ModelConfig):
        self.users_df = users_df
        self.output = None
        self.model_config = model_config

    @timeit
    def process_for_training(self):
        self.output = self.users_df[self.model_config.data_user_features]
        return self

    @timeit
    def process_for_inference(self):
        self.output = self.users_df[self.model_config.data_user_features]
        return self
