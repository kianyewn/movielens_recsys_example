class DataProcessingConfig:
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
