class LGBMConfig:
    def __init__(self):
        self.num_negative_samples = 5
        self.data_user_features = ["user_id", "gender", "age", "occupation", "zip"]
        self.data_movie_features = ["movie_id", "title", "year"]
        self.model_user_features = ["user_id", "gender", "age", "occupation", "zip"]
        self.model_movie_features = ["movie_id", "year"]

        # Users Data Type
        self.model_user_numerical_features = ["age"]
        self.model_user_categorical_features = ["gender", "occupation"]
        self.final_model_user_features = (
            self.model_user_categorical_features + self.model_user_numerical_features
        )
        # movie data type
        self.model_movie_numerical_features = []
        self.model_movie_categorical_features = ["year"]
        self.final_model_movie_features = (
            self.model_movie_categorical_features + self.model_movie_numerical_features
        )

        # sample for test
        self.n_test_users = 100
        self.random_state = 42

        self.label_column = "label"
        self.qid_column = "qid"

        self.eval_metrics = "map"

        # Model Config
        self.mlflow_experiment_name = "LGBMRanker_Optimization"
        self.mlflow_run_name = "LGBMRanker_Tuning"
        self.optuna_direction = "maximize"
        self.optuna_mode = 'n_trials'
        self.n_trials = 20
        # self.optuna_mode = "sampler"  #  "n_trials"
        self.optuna_duration = 30

        self.n_estimators = 200
        self.early_stopping_rounds = 100
        self.random_state = 42
        self.verbose = -1

        self.best_model_path = "artefacts/models/lgbm_best_model.pkl"
        self.default_model_path = "artefacts/models/lgbm_default_model.pkl"
        self.results_path = "artefacts/evaluations/lgbm_results.csv"

        self.inference_top_k = 3

    def to_dict(self):
        return self.__dict__
