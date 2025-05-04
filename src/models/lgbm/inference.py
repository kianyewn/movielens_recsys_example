import joblib
import pandas as pd

from configs.lgbm_config import LGBMConfig
from src.utils.utils import plot_learning_curve_plotly


class InferenceEngine:
    def __init__(self, model_config: LGBMConfig):
        self.model = None
        self.model_config = model_config
        self.features = (
            model_config.final_model_user_features
            + model_config.final_model_movie_features
        )
        self.label_col = model_config.label_column
        self.qid_col = model_config.qid_column

        self.load_model()

    def load_model(self):
        self.model = joblib.load(self.model_config.best_model_path)
        self.default_model = joblib.load(self.model_config.default_model_path)

    def prepare_data(self, inference_data: pd.DataFrame):
        self.master_data = inference_data.sort_values(by=self.qid_col).reset_index(
            drop=True
        )
        self.X_train = self.master_data[self.features]
        self.y_train = self.master_data[self.label_col]
        self.train_groups = self.master_data[self.qid_col].value_counts().sort_index()

    def retrain(self, inference_data: pd.DataFrame):
        self.prepare_data(inference_data)
        self.model.fit(
            X=self.X_train,
            y=self.y_train,
            group=self.train_groups,
            eval_set=[(self.X_train, self.y_train)],
            eval_group=[self.train_groups],
            eval_metric=self.data_config.eval_metrics,
        )

        plot_learning_curve_plotly(
            self.model,
            metrics_to_plot=self.data_config.eval_metrics,
            legend_labels=["train"],
        )
        return self

    def predict(self, test):
        return self.model.predict(test[self.features])

    def predict_default(self, test):
        return self.default_model.predict(test[self.features])

    def score(self, inference_data: pd.DataFrame):
        inference_data["pred"] = self.predict(inference_data)
        inference_data["rank"] = inference_data.groupby("user_id")["pred"].rank(
            method="first", ascending=False
        )
        self.inference_data = inference_data[
            inference_data["rank"] <= self.model_config.inference_top_k
        ].sort_values(by=["user_id", "rank"])
        return self
