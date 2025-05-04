import numpy as np
import pandas as pd
from loguru import logger

from configs.processing_config import DataProcessingConfig
from src.utils.utils import timeit


class ProcessRatings:
    def __init__(
        self, ratings_df: pd.DataFrame, processing_config: DataProcessingConfig
    ):
        self.ratings_df = ratings_df
        self.processing_config = processing_config
        self.label_counts = None
        self.output = None

        self.user_pos_interactions = None
        self.user_neg_interactions = None

    def prepare_columns_for_ranking(self):
        logger.info("Preparing columns for ranking")
        # Rank interactions
        self.ratings_df["interaction_num"] = self.ratings_df.groupby("user_id")[
            "timestamp"
        ].transform("rank", method="first", ascending=False)
        # Rank interactions
        self.ratings_df["qid"] = np.arange(self.ratings_df.shape[0])
        logger.info("Completed preparing columns for ranking")
        return self

    def negative_sampling(self):
        logger.info("Preparing Negative Sampling")
        # prepare negatives
        movies_set = set(self.ratings_df["movie_id"].unique())
        user_pos_interactions = (
            self.ratings_df.groupby("user_id")["movie_id"].apply(set).to_dict()
        )
        user_neg_interactions = {
            uid: list(movies_set - pos_set)
            for uid, pos_set in user_pos_interactions.items()
        }
        self.user_pos_interactions = user_pos_interactions
        self.user_neg_interactions = user_neg_interactions

        self.ratings_df["negative_samples"] = self.ratings_df["user_id"].apply(
            lambda x: np.random.choice(
                user_neg_interactions[x],
                size=self.processing_config.num_negative_samples,
                replace=False,
            )
        )
        ratings_neg = self.ratings_df.explode("negative_samples").copy()
        ratings_neg["movie_id"] = ratings_neg["negative_samples"]
        ratings_neg = ratings_neg.drop("negative_samples", axis=1)
        ratings_neg["label"] = 0
        logger.info("Joining Positive and Negative Samples")
        # combine with postives
        ratings_pos = self.ratings_df.drop("negative_samples", axis=1)
        ratings_pos["label"] = 1
        ratings_pos_neg = pd.concat([ratings_pos, ratings_neg])
        self.label_counts = ratings_pos_neg["label"].value_counts()

        self.output = ratings_pos_neg
        logger.info("completed.")
        return self

    @timeit
    def process_for_training(self):
        self.prepare_columns_for_ranking()
        self.negative_sampling()
        return self

    @timeit
    def process_for_inference(self):
        self.prepare_columns_for_ranking()
        self.output = self.ratings_df
        return self
