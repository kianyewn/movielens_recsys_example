import pandas as pd
from loguru import logger

from configs.model_config import ModelConfig
from src.feature_processing.encoders import CategoricalEncoder, NumericalEncoder
from src.utils.utils import timeit, value_counts


class ProcessFeatures:
    def __init__(
        self,
        users: pd.DataFrame,
        movies: pd.DataFrame,
        ratings: pd.DataFrame,
        model_config: ModelConfig,
    ):
        self.users = users
        self.movies = movies
        self.ratings = ratings
        self.model_config = model_config
        self.u2id = None
        self.id2u = None
        self.m2id = None
        self.id2m = None
        self.master_data = None
        self.train_label_value_counts = None
        self.val_label_value_counts = None
        self.test_label_value_counts = None

        self.numerical_encoder = None
        self.categorical_encoder = None

    def combine_features(self):
        logger.info("Combining features to Ratings")
        users_feat = self.users[self.model_config.model_user_features]
        movies_feat = self.movies[self.model_config.model_movie_features]

        master_data = self.ratings.merge(users_feat, on="user_id", how="left").merge(
            movies_feat, on="movie_id", how="left"
        )

        # as if user_id are strings, like in production
        master_data["user_id_str"] = master_data["user_id"].astype(str)
        master_data["movie_id_str"] = master_data["movie_id"].astype(str)

        self.u2id = {
            u: i + 1 for i, u in enumerate(sorted(master_data["user_id_str"].unique()))
        }
        self.id2u = {i: u for u, i in self.u2id.items()}
        self.m2id = {
            m: i + 1 for i, m in enumerate(sorted(master_data["movie_id_str"].unique()))
        }
        self.id2m = {i: m for m, i in self.m2id.items()}

        master_data["user_id"] = master_data["user_id_str"].map(self.u2id)
        master_data["movie_id"] = master_data["movie_id_str"].map(self.m2id)

        self.master_data = master_data
        logger.info("Completed combining features to Ratings")
        return self

    def get_inference_scoring_data(self):
        top_k = self.model_config.inference_retrieved_top_k_popular_movies
        top_100_popular_movies = pd.DataFrame(
            self.movies["movie_id"].value_counts().index[:top_k], columns=["movie_id"]
        )
        top_100_popular_movies["movie_id_str"] = top_100_popular_movies["movie_id"].map(
            self.id2m
        )
        top_100_popular_movies["dummy_col"] = 1

        inference_users = self.users[["user_id"]].drop_duplicates()
        inference_users["user_id_str"] = inference_users["user_id"].map(self.id2u)
        inference_users["dummy_col"] = 1

        inference_data = inference_users.merge(
            top_100_popular_movies, on="dummy_col", how="left"
        )
        inference_data = inference_data.drop(columns=["dummy_col"], axis=1)

        users_feat = self.users[self.model_config.model_user_features]
        movies_feat = self.movies[self.model_config.model_movie_features]
        inference_data = inference_data.merge(
            users_feat, on="user_id", how="left"
        ).merge(movies_feat, on="movie_id", how="left")
        self.inference_data = inference_data
        return self

    def train_test_split(self):
        logger.info("Creating train and val data for modelling")
        val = self.master_data[self.master_data["interaction_num"] == 1]
        train = self.master_data[self.master_data["interaction_num"] != 1]
        test_users = (
            val["user_id"]
            .drop_duplicates()
            .sample(
                self.model_config.n_test_users,
                random_state=self.model_config.random_state,
            )
        )
        test = val[val["user_id"].isin(test_users)].reset_index(drop=True)
        val = val[~val["user_id"].isin(test_users)].reset_index(drop=True)

        self.val_label_value_counts = value_counts(val, "label")
        self.train_label_value_counts = value_counts(train, "label")
        self.test_label_value_counts = value_counts(test, "label")
        self.val_data = val
        self.train_data = train
        self.test_data = test
        logger.info("Completed creating train and val data for modelling")
        return self

    def numerical_feature_processing_for_training(self):
        logger.info("Processing numerical features")
        self.numerical_encoder = NumericalEncoder(
            self.model_config.model_user_numerical_features
            + self.model_config.model_movie_numerical_features
        )
        self.numerical_encoder.fit(self.train_data)
        self.train_data = self.numerical_encoder.transform(self.train_data)
        self.val_data = self.numerical_encoder.transform(self.val_data)
        self.test_data = self.numerical_encoder.transform(self.test_data)
        logger.info("Completed processing numerical features")
        return self

    def categorical_feature_processing_for_training(self):
        logger.info("Processing categorical features")
        self.categorical_encoder = CategoricalEncoder(
            self.model_config.model_user_categorical_features
            + self.model_config.model_movie_categorical_features
        )

        self.categorical_encoder.fit(self.train_data)
        self.train_data = self.categorical_encoder.transform(self.train_data)
        self.val_data = self.categorical_encoder.transform(self.val_data)
        self.test_data = self.categorical_encoder.transform(self.test_data)
        logger.info("Completed processing categorical features")
        return self

    def numerical_feature_processing_for_inference(self):
        logger.info("Processing numerical features")
        self.inference_data = self.numerical_encoder.transform(self.inference_data)
        logger.info("Completed processing numerical features")
        return self

    def categorical_feature_processing_for_inference(self):
        logger.info("Processing categorical features")
        self.inference_data = self.categorical_encoder.transform(self.inference_data)
        logger.info("Completed processing categorical features")
        return self

    @timeit
    def process_for_training(self):
        self.combine_features()
        self.train_test_split()
        self.numerical_feature_processing_for_training()
        self.categorical_feature_processing_for_training()
        return self

    @timeit
    def process_for_inference(self):
        self.get_inference_scoring_data()
        self.numerical_feature_processing_for_inference()
        self.categorical_feature_processing_for_inference()
        return self

    def save(self):
        from joblib import dump

        attributes_to_save = {
            "model_config": self.model_config,
            "u2id": self.u2id,
            "id2u": self.id2u,
            "m2id": self.m2id,
            "id2m": self.id2m,
            "master_data": self.master_data,
            "train_label_value_counts": self.train_label_value_counts,
            "val_label_value_counts": self.val_label_value_counts,
            "test_label_value_counts": self.test_label_value_counts,
            "numerical_encoder": self.numerical_encoder,
            "categorical_encoder": self.categorical_encoder,
        }
        dump(attributes_to_save, self.model_config.process_features_object_path)

    def load(self):
        from joblib import load

        attributes_loaded = load(self.model_config.process_features_object_path)

        self.model_config = self.model_config.update(attributes_loaded["model_config"])
        self.u2id = attributes_loaded["u2id"]
        self.id2u = attributes_loaded["id2u"]
        self.m2id = attributes_loaded["m2id"]
        self.id2m = attributes_loaded["id2m"]
        self.master_data = attributes_loaded["master_data"]
        self.train_label_value_counts = attributes_loaded["train_label_value_counts"]
        self.val_label_value_counts = attributes_loaded["val_label_value_counts"]
        self.test_label_value_counts = attributes_loaded["test_label_value_counts"]
        self.numerical_encoder = attributes_loaded["numerical_encoder"]
        self.categorical_encoder = attributes_loaded["categorical_encoder"]

        logger.info("Loaded")
        return self
