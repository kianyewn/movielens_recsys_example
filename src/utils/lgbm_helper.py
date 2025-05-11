import pandas as pd
from matplotlib import pyplot as plt


def get_lgbm_feature_importance_df(model, importance_types=["split", "gain"]):
    feature_importane_dfs = []
    for importance_type in importance_types:
        feature_importance_dict = {"importance_type": importance_type}
        feature_importance_dict["feature_importance"] = (
            model.booster_.feature_importance(importance_type=importance_type)
        )
        feature_importance_dict["feature"] = model.feature_name_
        feature_importance_df = pd.DataFrame(feature_importance_dict).sort_values(
            by="feature_importance", ascending=False
        )
        feature_importance_df["rank"] = feature_importance_df[
            "feature_importance"
        ].rank(method="first", ascending=False)
        feature_importane_dfs.append(feature_importance_df)
    feature_importance_df = pd.concat(feature_importane_dfs, axis=0)
    return feature_importance_df


class LGBMExplainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.features = self.config.features
        self.qid_col = self.config.qid_column
        self.label_col = self.config.label_column

        self.fi_df = None

    # def plot_feature_importance(self, top_k=10, importance_type='split'):
    #     if self.fi_df is None:
    #         self.fi_df = get_lgbm_feature_importance_df(self.model)
    #     fi_df = self.fi_df
    #     fi_df = fi_df[(fi_df['rank'] <= top_k) & (fi_df['importance_type'] == importance_type)]
    #     fi_df = fi_df.sort_values(by='feature_importance', ascending=True)
    #     fig, ax = plt.subplots()
    #     ax.barh(fi_df['feature'], fi_df['feature_importance'])
    #     plt.show()

    def plot_feature_importance(
        self,
        top_k=10,
        importance_type="split",
        figsize=(10, 8),
        color="C0",
        title="Top Feature Importance",
        save_path=None,
    ):
        """
        Plot feature importance with clear annotations and formatting.

        Parameters:
        -----------
        top_k : int
            Number of top features to display
        figsize : tuple
            Figure size (width, height)
        color : str
            Bar color
        title : str
            Plot title
        save_path : str, optional
            Path to save the figure

        Returns:
        --------
        fig : matplotlib figure
        """
        if self.fi_df is None:
            self.fi_df = get_lgbm_feature_importance_df(self.model)

        # Get top k features
        fi_df = self.fi_df.copy()
        fi_df = fi_df[
            (fi_df["rank"] <= top_k) & (fi_df["importance_type"] == importance_type)
        ]
        fi_df = fi_df.sort_values(by="feature_importance", ascending=True)

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Plot horizontal bars
        bars = ax.barh(
            fi_df["feature"],
            fi_df["feature_importance"],
            color=color,
            edgecolor="gray",
            alpha=0.8,
        )

        # Add value annotations to each bar
        for bar in bars:
            width = bar.get_width()
            ax.text(
                width + (width * 0.02),  # Slightly offset from end of bar
                bar.get_y() + bar.get_height() / 2,
                f"{width:.4f}",
                va="center",
                fontsize=9,
            )

        # Add labels and title
        ax.set_xlabel("Feature Importance", fontsize=12)
        ax.set_ylabel("Feature", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")

        # Add grid lines for better readability
        ax.grid(axis="x", linestyle="--", alpha=0.6)

        # Improve y-axis labels (feature names)
        ax.tick_params(axis="y", labelsize=10)

        # Add a note about the importance type
        ax.text(
            0.01,
            -0.1,
            f"Importance type: {importance_type}",
            transform=ax.transAxes,
            fontsize=9,
            style="italic",
        )

        # Add model name if available
        model_name = self.model.__class__.__name__
        plt.figtext(
            0.99, 0.01, f"Model: {model_name}", ha="right", fontsize=8, style="italic"
        )

        # Adjust layout
        plt.tight_layout()

        # Save figure if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def inspect_important_features(self, top_k=10, importance_type="split"):
        if self.fi_df is None:
            self.fi_df = get_lgbm_feature_importance_df(self.model)
        fi_df = self.fi_df.copy()
        fi_df = fi_df[
            (fi_df["rank"] <= top_k) & (fi_df["importance_type"] == importance_type)
        ]
        fi_df = fi_df.sort_values(by="feature_importance", ascending=False)
        fi_df["feature_type"] = fi_df["feature"].apply(
            lambda x: "categorical"
            if x in self.config.categorical_features
            else "numerical"
        )
        return fi_df

    def inspect_important_numerical_features(self, top_k=10, importance_type="split"):
        if self.fi_df is None:
            self.fi_df = get_lgbm_feature_importance_df(self.model)
        fi_df = self.fi_df.copy()
        fi_df = fi_df[
            (fi_df["importance_type"] == importance_type)
            & (fi_df["feature"].isin(self.config.numerical_features))
        ]
        fi_df = fi_df.sort_values(by="feature_importance", ascending=False).iloc[:top_k]
        fi_df["feature_type"] = fi_df["feature"].apply(
            lambda x: "categorical"
            if x in self.config.categorical_features
            else "numerical"
        )
        return fi_df

    def inspect_important_categorical_features(self, top_k=10, importance_type="split"):
        if self.fi_df is None:
            self.fi_df = get_lgbm_feature_importance_df(self.model)
        fi_df = self.fi_df.copy()
        fi_df = fi_df[
            (fi_df["rank"] <= top_k)
            & (fi_df["importance_type"] == importance_type)
            & (fi_df["feature"].isin(self.config.categorical_features))
        ]
        fi_df = fi_df.sort_values(by="feature_importance", ascending=False)
        fi_df["feature_type"] = fi_df["feature"].apply(
            lambda x: "categorical"
            if x in self.config.categorical_features
            else "numerical"
        )
        return fi_df

    def inspect_importance_feature_names_by_dtype(
        self, top_k=10, importance_type="split"
    ):
        categorical_features = self.inspect_important_categorical_features(
            top_k, importance_type
        )
        numerical_features = self.inspect_important_numerical_features(
            top_k, importance_type
        )
        fi_df = (
            pd.concat([categorical_features, numerical_features])
            .sort_values(by=["feature_type", "feature_importance"], ascending=False)
            .reset_index(drop=True)
        )
        fi_df = fi_df[
            ["importance_type", "feature_type", "feature", "feature_importance", "rank"]
        ]
        return fi_df

    def get_important_feature_names(self, top_k=10, importance_type="split"):
        if self.fi_df is None:
            self.fi_df = get_lgbm_feature_importance_df(self.model)
        fi_df = self.fi_df.copy()
        fi_df = fi_df[
            (fi_df["rank"] <= top_k) & (fi_df["importance_type"] == importance_type)
        ]
        fi_df = fi_df.sort_values(by="feature_importance", ascending=False)
        return fi_df["feature"].tolist()

    def get_important_categorical_feature_names(
        self, top_k=10, importance_type="split"
    ):
        if self.fi_df is None:
            self.fi_df = get_lgbm_feature_importance_df(self.model)
        fi_df = self.fi_df.copy()
        fi_df = fi_df[
            (fi_df["importance_type"] == importance_type)
            & (fi_df["feature"].isin(self.config.categorical_features))
        ]
        fi_df = fi_df.sort_values(by="feature_importance", ascending=False).iloc[:top_k]
        return fi_df["feature"].tolist()

    def get_important_numerical_feature_names(self, top_k=10, importance_type="split"):
        if self.fi_df is None:
            self.fi_df = get_lgbm_feature_importance_df(self.model)
        fi_df = self.fi_df.copy()
        fi_df = fi_df[
            (fi_df["importance_type"] == importance_type)
            & (fi_df["feature"].isin(self.config.numerical_features))
        ]
        fi_df = fi_df.sort_values(by="feature_importance", ascending=False).iloc[:top_k]
        return fi_df["feature"].tolist()


# Example usage:
# self.plot_feature_importance(top_k=15, figsize=(12, 10),
#                             title='LightGBM Feature Importance (Top 15)',
#                             save_path='feature_importance.png')

# lgbm_config = LGBMConfig()
# lgbm_exp = LGBMExplainer(tuner.best_model, lgbm_config)
# fig = lgbm_exp.plot_feature_importance(top_k=5, importance_type='split',figsize=(10, 5),)
# important_features = lgbm_exp.get_important_feature_names(top_k=5, importance_type='split')
# important_categorical_features = lgbm_exp.get_important_categorical_feature_names(top_k=5, importance_type='split')
# important_numerical_features = lgbm_exp.get_important_numerical_feature_names(top_k=5, importance_type='split')
# lgbm_exp.inspect_importance_feature_names_by_dtype(top_k=5, importance_type='split')


import time
from dataclasses import dataclass

import joblib
import mlflow
import optuna
import pandas as pd
import torch
import torchmetrics
from lightgbm import LGBMRanker
from loguru import logger
from torchmetrics.retrieval.base import RetrievalMetric

from configs.lgbm_config import LGBMConfig
from src.evaluation.ranking_evaluator import TopKRankingEvaluator
from src.utils.utils import plot_learning_curve_plotly

# Set the tracking URI to the local directory
mlflow.set_tracking_uri("file:./mlruns")


class BaseOptunaTuner:
    def __init__(self, config):
        self.config = config
        self.trial_df = None

    def get_trials_dataframe(self):
        if self.study is None:
            raise ValueError("Study is not initialized")
        self.trial_df = self.study.trials_dataframe()
        return self.trial_df

    def get_parameter_from_trial_number(self, trial_number):
        trial_df = self.get_trials_dataframe()
        param_cols = [col for col in trial_df.columns if "params_" in col]
        single_trial_params = trial_df[trial_df["number"] == trial_number][
            param_cols
        ].to_dict(orient="records")[0]
        params = {k.replace("params_", ""): v for k, v in single_trial_params.items()}
        return params

    def create_study(self):
        if self.config.optuna_mode == "n_trials":
            study = optuna.create_study(
                direction=self.config.optuna_objective_direction
            )
        elif self.model_config.optuna_mode == "sampler":
            sampler = optuna.samplers.TPESampler(seed=self.config.random_state)
            study = optuna.create_study(
                direction=self.config.optuna_objective_direction, sampler=sampler
            )
        else:
            raise ValueError(f"Invalid optuna mode: {self.config.optuna_mode}")
        return study

    def optimize(self, study):
        if (
            self.config.optuna_duration is not None
            and self.config.optuna_mode == "sampler"
        ):
            tic = time.time()
            while time.time() - tic < self.config.optuna_duration:
                study.optimize(self.objective, n_trials=1)
        else:
            logger.info(f"Optimizing for {self.config.optuna_n_trials} trials")
            study.optimize(self.objective, n_trials=self.config.optuna_n_trials)

    def objective(self):
        pass

    def tune(self):
        pass

    def get_final_best_params(self):
        pass


@dataclass
class RankingScorer:
    name: str
    scorer: RetrievalMetric

    def score(self, preds, targets, indexes):
        if not isinstance(preds, torch.Tensor):
            preds = torch.tensor(preds)
        if not isinstance(targets, torch.Tensor):
            targets = torch.tensor(targets)
        if not isinstance(indexes, torch.Tensor):
            indexes = torch.tensor(indexes)

        score = self.scorer(preds, targets, indexes)
        return score


class LGBMRankerOptunaTuner(BaseOptunaTuner):
    def __init__(self, train_df, valid_df, model_config, scorer: RankingScorer):
        self.train_df = train_df
        self.valid_df = valid_df
        self.scorer = scorer
        self.config = model_config
        self.features = self.config.features
        self.qid_col = self.config.qid_column
        self.label_col = self.config.label_column

        self.X_train = None
        self.y_train = None
        self.train_groups = None
        self.X_val = None
        self.y_val = None
        self.val_groups = None

        self.experiment_name = self.config.mlflow_experiment_name
        self.experiment_id = None

        self.evaluator = None
        self.default_model = None

        self.prepare_data()

    def prepare_data(self):
        # important to sort by qid for LGBMRanker
        self.train_df = self.train_df.sort_values(by=self.qid_col).reset_index(
            drop=True
        )
        self.valid_df = self.valid_df.sort_values(by=self.qid_col).reset_index(
            drop=True
        )

        self.X_train = self.train_df[self.features]
        self.y_train = self.train_df[self.label_col]
        self.train_groups = self.train_df[self.qid_col].value_counts().sort_index()
        self.X_val = self.valid_df[self.features]
        self.y_val = self.valid_df[self.label_col]
        self.val_groups = self.valid_df[self.qid_col].value_counts().sort_index()
        return

    @property
    def fixed_params(self):
        params = {
            "n_estimators": self.config.n_estimators,
            "early_stopping_round": self.config.early_stopping_round,
            "random_state": self.config.random_state,
            "verbose": self.config.verbose,
        }
        return params

    @property
    def fit_params(self):
        params = {
            "group": self.train_groups,
            "eval_set": [(self.X_train, self.y_train), (self.X_val, self.y_val)],
            "eval_group": [self.train_groups, self.val_groups],
            "eval_metric": self.config.eval_metrics,
            "eval_at": self.config.eval_at,
        }
        return params

    # Define Optuna objective function
    def objective(self, trial):
        # Start MLflow nested run
        with mlflow.start_run(nested=True):
            # Define hyperparameters to optimize
            params = {
                "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt"]),
                "num_leaves": trial.suggest_int("num_leaves", 20, 30),
                "max_depth": trial.suggest_int("max_depth", 2, 10),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                ),
                # 'subsample_for_bin': trial.suggest_int('subsample_for_bin', 50000, 300000),
                "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.2),
                "min_child_weight": trial.suggest_float(
                    "min_child_weight", 0.001, 0.1, log=True
                ),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "subsample_freq": trial.suggest_int("subsample_freq", 0, 10),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "objective": trial.suggest_categorical(
                    "objective", ["lambdarank", "rank_xendcg"]
                ),
            }

            # Special handling for GOSS boosting
            if params["boosting_type"] == "goss":
                params["subsample"] = 1.0  # GOSS doesn't use traditional subsampling

            # Log parameters to MLflow
            mlflow.log_params(params)

            # Create and train model
            model = LGBMRanker(
                **self.fixed_params,
                **params,
            )

            # Track training time
            start_time = time.time()

            # Fit model with early stopping
            model.fit(self.X_train, self.y_train, **self.fit_params)

            training_time = time.time() - start_time

            # Get predictions
            val_preds = model.predict(self.X_val)
            val_targets = self.y_val
            val_indexes = self.valid_df[self.qid_col]

            # calcualte metrics
            score = self.scorer.score(val_preds, val_targets, val_indexes)

            # Log metrics to MLflow
            mlflow.log_metric(f"{self.scorer.name}", score)
            mlflow.log_metric("training_time", training_time)

            # Store best iteration
            trial.set_user_attr("best_iteration", model.best_iteration_)
            return score

    def train_best_model(self, best_params):
        fixed_params = self.fixed_params
        model = LGBMRanker(**fixed_params, **best_params)
        model.fit(self.X_train, self.y_train, **self.fit_params)
        return model

    def best_model_learning_curve(self, best_model):
        eval_results = best_model.evals_result_
        eval_dfs = []
        for eval_set_type in eval_results:
            eval_df = pd.DataFrame(eval_results[eval_set_type]).add_prefix(
                f"{eval_set_type}_"
            )
            eval_dfs.append(eval_df)

        eval_results_df = pd.concat(eval_dfs, axis=1).reset_index(names=["iteration"])
        return eval_results_df

    def tune(self):
        experiment_name = self.experiment_name
        try:
            experiment_id = mlflow.create_experiment(experiment_name)
        except Exception as _:
            experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

        self.experiment_id = experiment_id

        # Run the experiment
        with mlflow.start_run(
            experiment_id=self.experiment_id, run_name=self.experiment_name
        ):
            self.study = self.create_study()
            self.optimize(self.study)

            best_params = self.study.best_params
            best_value = self.study.best_value

            best_iteration = self.study.best_trial.user_attrs["best_iteration"]
            best_params_msg = {f"best_{k}": v for k, v in best_params.items()}
            # best_params.update({"best_iteration": best_iteration})
            mlflow.log_params(best_params_msg)
            mlflow.log_metric(f"best_{self.scorer.name}", best_value)

            # Train final model with best parameters
            self.best_model = self.train_best_model(best_params)
            self.best_iteration = best_iteration
            self.best_params = best_params

            # Get predictions
            val_preds = self.best_model.predict(self.X_val)
            val_targets = self.y_val
            val_indexes = self.valid_df[self.qid_col]

            # calcualte metrics
            final_score = self.scorer.score(val_preds, val_targets, val_indexes)
            mlflow.log_metric(f"final_{self.scorer.name}", final_score)

            logger.info(f"Best {self.scorer.name}: {best_value:.4f}")
            logger.info(f"Final {self.scorer.name}: {final_score:.4f}")
            logger.info(
                f"Best parameters: {best_params_msg}, best_iteration: {self.best_iteration}"
            )

            self.trials_df = self.get_trials_dataframe()

            plot_learning_curve_plotly(
                self.best_model,
                metrics_to_plot=self.config.eval_metrics,
                legend_labels=["train", "validation"],
            )

        mlflow.end_run()
        return self.best_model

    def predict(self, data):
        data["pred"] = self.best_model.predict(data[self.features])
        return data

    def init_evaluator(self):
        from src.evaluation.ranking_evaluator import TopKRankingEvaluator

        self.evaluator = TopKRankingEvaluator(list_k=[1, 2, 3, 4, 5])
        return

    def evaluate(self, data):
        if self.evaluator is None:
            self.init_evaluator()
        data["pred"] = self.best_model.predict(data[self.features])
        eval_df = self.evaluator.evaluate(data)
        # # Log metrics
        # for metric_name, metric_dict in eval_df.set_index('k').to_dict().items():
        #     for k, value in metric_dict.items():
        #         # Format as "k1_ndcg", "k5_precision", etc.
        #         print(f"{metric_name}_at_{k}", value)
        return eval_df

    def evaluate_model(self, model, data):
        if self.evaluator is None:
            self.init_evaluator()
        data["pred"] = model.predict(data[self.features])
        eval_df = self.evaluator.evaluate(data)
        return eval_df

    def train_final_model(self, full_data):
        logger.info("Full training")
        best_params = self.best_params.copy()
        fixed_params = self.fixed_params.copy()
        fixed_params.pop("early_stopping_round", None)
        fixed_params.update({"n_estimators": self.best_iteration})
        self.final_model = LGBMRanker(**fixed_params, **best_params)

        full_data = full_data.sort_values(by=self.qid_col).reset_index(drop=True)
        full_X = full_data[self.features]
        full_y = full_data[self.label_col]
        full_group = full_data[self.qid_col].value_counts().sort_index()
        self.final_model.fit(full_X, full_y, group=full_group)
        # eval_set=[(full_X, full_y)], eval_group=[full_group], eval_metric=self.config.eval_metrics, eval_at=self.config.eval_at)

        # full_trained_eval_df = self.evaluate_model(self.final_model, full_data)
        return self.final_model

    def get_evaluation_results_from_trial_number(self, trial_number):
        trial_params = self.get_parameter_from_trial_number(trial_number)
        trial_model = LGBMRanker(**self.fixed_params, **trial_params)
        trial_model.fit(self.X_train, self.y_train, **self.fit_params)
        eval_result = self.evaluate_model(trial_model, self.valid_df)
        return eval_result

    def evaluate_default_model(self, val):
        if self.default_model is None:
            self.default_model = LGBMRanker(random_state=self.config.random_state)
            self.default_model.fit(self.X_train, self.y_train, group=self.train_groups)

        default_eval_result = self.evaluate_model(self.default_model, val)
        return default_eval_result

    def get_trained_vs_default_eval_results(self, data, title="validation"):
        print(f"{title} evaluation results")
        valid_df_eval_result = self.evaluate(data)
        default_eval_result = self.evaluate_default_model(data)
        # valid_df_eval_result, default_eval_result = self.get_trained_vs_default_eval_results(data)
        default_eval_result = default_eval_result.set_index("k").add_prefix("default_")
        valid_df_eval_result = valid_df_eval_result.set_index("k").add_prefix("model_")
        eval_result = pd.concat([default_eval_result, valid_df_eval_result], axis=1)
        eval_result = eval_result[
            sorted(eval_result.columns, key=lambda x: x.split("_", 1)[1])
        ]

        return valid_df_eval_result, default_eval_result, eval_result


# from configs.lgbm_config import LGBMConfig
# lgbm_config = LGBMConfig()
# lgbm_config['mlflow_experiment_name'] = "LGBMRanker_Optimization_narrower"
# # print(lgbm_config['mlflow_experiment_name'])
# ndcg_at_5 = RankingScorer(name="ndcg_at_5", scorer=RetrievalNormalizedDCG(top_k=5))
# # tuner = LGBMRankerOptunaTuner(train, val, lgbm_config, ndcg_at_5)
# # best_moodel = tuner.tune()
# # tuner.get_trained_vs_default_eval_results = LGBMRankerOptunaTuner.get_trained_vs_default_eval_results
# valid_df_eval_result, default_eval_result, eval_result = tuner.get_trained_vs_default_eval_results(val, title='validation')
# final_results = tuner.train_final_model(pd.concat([train, val, test]))
# tuner.final_model
