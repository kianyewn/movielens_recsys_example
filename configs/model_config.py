class ModelConfig:
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

        # self.eval_metrics = ['ndcg@2','ndcg@3','ndcg@4','ndcg@5']
        # self.eval_metrics = ['map@2','map@3','map@4','map@5']
        self.eval_metrics = "map"

        # Model Config
        self.mlflow_experiment_name = "LGBMRanker_Optimization"
        self.mlflow_run_name = "LGBMRanker_Tuning"
        self.n_trials = 5
        self.optuna_direction = "maximize"
        self.optuna_mode = "sampler"  #  "n_trials"
        self.optuna_duration = 30

        self.n_estimators = 200
        self.early_stopping_rounds = 100
        self.random_state = 42
        self.verbose = -1

        # Modelling Artefacts
        self.process_features_object_path = (
            "artefacts/models/process_features_object.joblib"
        )

        # Inference
        self.inference_retrieved_top_k_popular_movies = 10

    def update(self, other):
        for key, value in other.__dict__.items():
            setattr(self, key, value)
        return self
