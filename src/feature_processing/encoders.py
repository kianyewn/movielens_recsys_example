from loguru import logger
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


class NumericalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, numerical_columns):
        self.numerical_columns = numerical_columns
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X[self.numerical_columns])
        return self

    def transform(self, X, y=None):
        X_transformed = X.copy()
        X_transformed[self.numerical_columns] = self.scaler.transform(
            X[self.numerical_columns]
        )
        self.feature_names_out = X.columns.tolist()
        return X_transformed

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out

    def save(self, path):
        """Save encoder attributes to a joblib file.

        Args:
            path: Path to save the joblib file
        """
        from joblib import dump

        attributes = {
            "numerical_columns": self.numerical_columns,
            "scaler": self.scaler,
            "feature_names_out": self.feature_names_out,
        }

        dump(attributes, path)
        logger.info(f"Saved encoder attributes to {path}")

    def load(self, path):
        """Load encoder attributes from a joblib file.

        Args:
            path: Path to the joblib file
        """
        from joblib import load

        attributes = load(path)

        self.numerical_columns = attributes["numerical_columns"]
        self.scaler = attributes["scaler"]
        self.feature_names_out = attributes["feature_names_out"]

        return self


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_columns):
        self.categorical_columns = categorical_columns
        self.encoding_dict = {}
        self.unknown_id = 0

    def fit(self, X, y=None):
        for col in self.categorical_columns:
            unique = X[col].value_counts(dropna=False).index.tolist()
            self.encoding_dict[col] = {v: i + 1 for i, v in enumerate(unique)}
        self.feature_names_out = X.columns.tolist()
        return self

    def transform(self, X, y=None):
        X_transformed = X.copy()
        for col in self.categorical_columns:
            X_transformed[col] = (
                X_transformed[col].map(self.encoding_dict[col]).fillna(self.unknown_id)
            )
        return X_transformed

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out

    def save(self, path):
        """Save encoder attributes to a joblib file.

        Args:
            path: Path to save the joblib file
        """
        from joblib import dump

        attributes = {
            "categorical_columns": self.categorical_columns,
            "encoding_dict": self.encoding_dict,
            "unknown_id": self.unknown_id,
            "feature_names_out": self.feature_names_out,
        }

        dump(attributes, path)

    def load(self, path):
        """Load encoder attributes from a joblib file.

        Args:
            path: Path to the joblib file
        """
        from joblib import load

        attributes = load(path)

        self.categorical_columns = attributes["categorical_columns"]
        self.encoding_dict = attributes["encoding_dict"]
        self.unknown_id = attributes["unknown_id"]
        self.feature_names_out = attributes["feature_names_out"]
        logger.info(f"Loaded encoder attributes from {path}")
        return self
