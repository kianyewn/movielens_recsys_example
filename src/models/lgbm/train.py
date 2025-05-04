import time

import joblib
import mlflow
import optuna
import pandas as pd
import torch
import torchmetrics
from lightgbm import LGBMRanker
from loguru import logger

from src.utils.utils import plot_learning_curve_plotly
from configs.lgbm_config import LGBMConfig
from src.evaluation.ranking_evaluator import RankingEvaluator

# Set the tracking URI to the local directory
mlflow.set_tracking_uri("file:./mlruns")

class TrainingEngine:
    def __init__(
        self,
        train,
        val,
        model_config: LGBMConfig,
        model=None,
        scorer: torchmetrics.retrieval = None,
        scorer_name="ndcg@10",
    ):
        self.model = model
        self.train_data = train
        self.valid_data = val
        self.model_config = model_config
        self.scorer = scorer
        self.scorer_name = scorer_name
        self.features = (
            model_config.final_model_user_features
            + model_config.final_model_movie_features
        )
        self.label_col = model_config.label_column
        self.qid_col = model_config.qid_column

        self.X_train = None
        self.y_train = None
        self.train_groups = None
        self.X_val = None
        self.y_val = None
        self.val_groups = None

        self.experiment_id = None
        self.experiment_name = self.model_config.mlflow_experiment_name

        self.prepare_data()

    def prepare_data(self):
        # important to sort by qid for LGBMRanker
        self.train_data = self.train_data.sort_values(by=self.qid_col).reset_index(
            drop=True
        )
        self.valid_data = self.valid_data.sort_values(by=self.qid_col).reset_index(
            drop=True
        )
        self.X_train = self.train_data[self.features]
        self.y_train = self.train_data[self.label_col]
        self.train_groups = self.train_data[self.qid_col].value_counts().sort_index()
        self.X_val = self.valid_data[self.features]
        self.y_val = self.valid_data[self.label_col]
        self.val_groups = self.valid_data[self.qid_col].value_counts().sort_index()

    def train(self):
        if self.model is None:
            logger.info(f"No model provided in class `{self.__class__.__name__}`.")
            raise ValueError("No model provided.")

        self.model.fit(
            X=self.X_train,
            y=self.y_train,
            group=self.train_groups,
            eval_set=[(self.X_train, self.y_train), (self.X_val, self.y_val)],
            eval_group=[self.train_groups, self.val_groups],
            eval_metric=self.data_config.eval_metrics,
        )

        plot_learning_curve_plotly(
            self.model,
            metrics_to_plot=self.data_config.eval_metrics,
            legend_labels=["train", "val"],
        )
        self.final_model = self.model
        return self

    def predict(self, test):
        if self.final_model is None:
            if self.model is None:
                raise ValueError("No model provided.")
            self.final_model = self.model
        return self.final_model.predict(test[self.features])

    # Define Optuna objective function
    def objective(self, trial):
        # Start MLflow nested run
        with mlflow.start_run(nested=True):
            # Define hyperparameters to optimize
            params = {
                "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt"]),
                "num_leaves": trial.suggest_int("num_leaves", 20, 50),
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
                n_estimators=self.model_config.n_estimators,  # fixed will use early stopping
                early_stopping_round=self.model_config.early_stopping_rounds,
                random_state=self.model_config.random_state,
                verbose=self.model_config.verbose,
                **params,
            )

            # Track training time
            start_time = time.time()

            # Fit model with early stopping
            model.fit(
                self.X_train,
                self.y_train,
                group=self.train_groups,
                eval_set=[(self.X_train, self.y_train), (self.X_val, self.y_val)],
                eval_metric=self.model_config.eval_metrics,
                eval_group=[self.train_groups, self.val_groups],
            )

            training_time = time.time() - start_time

            # Get predictions
            val_preds = model.predict(self.X_val)

            # calcualte metrics
            score = self.scorer(
                torch.tensor(val_preds),
                torch.tensor(self.y_val),
                torch.tensor(self.valid_data[self.qid_col]),
            )

            # Log metrics to MLflow
            mlflow.log_metric(f"{self.scorer_name }", score)
            mlflow.log_metric("training_time", training_time)

            # Store best iteration
            trial.set_user_attr("best_iteration", model.best_iteration_)
            return score

    def create_optuna_study(self):
        if self.model_config.optuna_mode == "n_trials":
            study = optuna.create_study(direction=self.model_config.optuna_direction)

        elif self.model_config.optuna_mode == "sampler":
            sampler = optuna.samplers.TPESampler(seed=self.model_config.random_state)
            study = optuna.create_study(
                direction=self.model_config.optuna_direction, sampler=sampler
            )
        return study

    def optimize_optuna(self, objective_func):
        if (
            self.model_config.optuna_duration is not None
            and self.model_config.optuna_mode == "sampler"
        ):
            tic = time.time()
            while time.time() - tic < self.model_config.optuna_duration:
                self.study.optimize(objective_func, n_trials=1)
        else:
            logger.info(f"Optimizing for {self.model_config.n_trials} trials")
            self.study.optimize(objective_func, n_trials=self.model_config.n_trials)

    def hyperparameter_tuning(self):
        # Set up MLflow experiment
        experiment_name = self.experiment_name
        try:
            experiment_id = mlflow.create_experiment(experiment_name)
        except Exception as e:
            experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
            logger.info(e)

        self.experiment_id = experiment_id
        # Run the optimization
        with mlflow.start_run(
            experiment_id=self.experiment_id, run_name=self.experiment_name
        ):
            # Configure the study
            self.study = self.create_optuna_study()

            # Run optimization
            self.optimize_optuna(self.objective)

            # Log best parameters and score
            best_params = self.study.best_params
            best_value = self.study.best_value

            best_iteration = self.study.best_trial.user_attrs["best_iteration"]
            best_params_msg = {f"best_{k}": v for k, v in best_params.items()}
            best_params_msg.update({"best_iteration": best_iteration})
            mlflow.log_params(best_params_msg)
            mlflow.log_metric(f"best_{self.scorer_name}", best_value)

            # Train final model with best parameters
            final_params = {
                "n_estimators": self.model_config.n_estimators,  # 400 ,  # More estimators for final model
                "early_stopping_round": self.model_config.early_stopping_rounds,
                "random_state": self.model_config.random_state,
                "verbose": self.model_config.verbose,
                **best_params,
            }
            self.final_params = final_params
            final_model = LGBMRanker(**final_params)
            final_model.fit(
                self.X_train,
                self.y_train,
                group=self.train_groups,
                eval_set=[(self.X_train, self.y_train), (self.X_val, self.y_val)],
                eval_metric=self.model_config.eval_metrics,
                eval_group=[self.train_groups, self.val_groups],
            )
            self.final_model = final_model
            # Log the final model
            # mlflow.xgboost.log_model(final_model, "xgb_ranker_model")

            # Make predictions on validation set
            self.valid_data['pred'] = self.final_model.predict(self.X_val)
            # Calculate NDCG@10
            final_score = self.scorer(
                torch.tensor(self.valid_data['pred']),
                torch.tensor(self.valid_data['label']),
                torch.tensor(self.valid_data[self.qid_col]),
            )

            # Evaluate best model
            evaluator = RankingEvaluator(list_k=[1, 2, 3, 4, 5])
            evaluator.evaluate(self.valid_data)
            self.best_model_results = pd.DataFrame(evaluator.metrics)

            mlflow.log_metric(f"final_{self.scorer_name}", final_score)

            logger.info(f"Best {self.scorer_name}: {best_value:.4f}")
            logger.info(f"Final {self.scorer_name}: {final_score:.4f}")
            logger.info(f"Best parameters: {best_params_msg}")

            plot_learning_curve_plotly(
                final_model,
                metrics_to_plot=self.model_config.eval_metrics,
                legend_labels=["train", "validation"],
            )

        mlflow.end_run()

    def train_best_model(self):
        best_params = self.final_params
        best_iteration = self.study.best_trial.user_attrs.get("best_iteration")
        best_params.update({"n_estimators": best_iteration})
        final_model = LGBMRanker(**best_params)

        full_data = pd.concat([self.train_data, self.valid_data]).reset_index(drop=True)
        full_data = full_data.sort_values(by=self.qid_col).reset_index(drop=True)
        full_groups = full_data[self.qid_col].value_counts().sort_index()
        full_X_train = full_data[self.features]
        full_y_train = full_data[self.label_col]

        final_model.fit(
            full_X_train,
            full_y_train,
            group=full_groups,
            eval_metric=self.model_config.eval_metrics,
            eval_set=[(full_X_train, full_y_train)],
            eval_group=[full_groups],
        )
        self.final_model = final_model
        # Log the final model
        # mlflow.xgboost.log_model(final_model, "xgb_ranker_model")

        plot_learning_curve_plotly(
            final_model,
            metrics_to_plot=self.model_config.eval_metrics,
            legend_labels=["train"],
        )

    def train_default_model(self):
        default_model = LGBMRanker()

        full_data = pd.concat([self.train_data, self.valid_data]).reset_index(drop=True)
        full_data = full_data.sort_values(by=self.qid_col).reset_index(drop=True)
        full_groups = full_data[self.qid_col].value_counts().sort_index()
        full_X_train = full_data[self.features]
        full_y_train = full_data[self.label_col]

        default_model.fit(
            full_X_train,
            full_y_train,
            group=full_groups,
            eval_metric=self.model_config.eval_metrics,
            eval_set=[(full_X_train, full_y_train)],
            eval_group=[full_groups],
        )
        self.default_model = default_model

        plot_learning_curve_plotly(
            default_model,
            metrics_to_plot=self.model_config.eval_metrics,
            legend_labels=["train"],
        )

    def save_best_model(self):
        joblib.dump(self.final_model, self.model_config.best_model_path)

    def save_default_model(self):
        joblib.dump(self.default_model, self.model_config.default_model_path)

    def save_models(self):
        self.save_best_model()
        self.save_default_model()
