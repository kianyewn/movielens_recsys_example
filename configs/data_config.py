from pathlib import Path
from typing import Optional, Union

from constants import ROOT_DIR


class ReadArgs:
    def __init__(self, **kwargs):
        self.init(**kwargs)

    def init(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self):
        return self.__dict__


class FileObject:
    def __init__(
        self, file_path: Union[str, Path], read_args: Optional[ReadArgs] = None
    ):
        self.file_path = file_path
        self.read_args = read_args.to_dict() if read_args else None


class DataConfig:
    def __init__(self):
        self.base_folder = ROOT_DIR / "data" / "training"
        self.raw_users = FileObject(
            file_path=self.base_folder / "01_raw_data" / "users.dat",
            read_args=ReadArgs(
                sep="::",
                engine="python",
                header=None,
                names=["user_id", "gender", "age", "occupation", "zip"],
            ),
        )
        self.raw_movies = FileObject(
            file_path=self.base_folder / "01_raw_data" / "movies.dat",
            read_args=ReadArgs(
                sep="::",
                engine="python",
                header=None,
                names=["movie_id", "title", "genres"],
                encoding="unicode_escape",
            ),
        )

        self.raw_ratings = FileObject(
            file_path=self.base_folder / "01_raw_data" / "ratings.dat",
            read_args=ReadArgs(
                sep="::",
                engine="python",
                header=None,
                names=["user_id", "movie_id", "rating", "timestamp"],
            ),
        )

        self.processed_users = FileObject(
            file_path=self.base_folder / "02_processed_data" / "users_processed.parquet"
        )
        self.processed_movies = FileObject(
            file_path=self.base_folder
            / "02_processed_data"
            / "movies_processed.parquet"
        )

        self.processed_ratings = FileObject(
            file_path=self.base_folder
            / "02_processed_data"
            / "processed_ratings.parquet"
        )

        self.train = FileObject(
            file_path=self.base_folder / "03_train_val_test_data" / "train.parquet"
        )
        self.val = FileObject(
            file_path=self.base_folder / "03_train_val_test_data" / "val.parquet"
        )

        self.test = FileObject(
            file_path=self.base_folder / "03_train_val_test_data" / "test.parquet"
        )


class InferenceDataConfig:
    def __init__(self):
        self.base_folder = ROOT_DIR / "data" / "inference"
        self.raw_users = FileObject(
            file_path=self.base_folder / "01_raw_data" / "users.dat",
            read_args=ReadArgs(
                sep="::",
                engine="python",
                header=None,
                names=["user_id", "gender", "age", "occupation", "zip"],
            ),
        )
        self.raw_movies = FileObject(
            file_path=self.base_folder / "01_raw_data" / "movies.dat",
            read_args=ReadArgs(
                sep="::",
                engine="python",
                header=None,
                names=["movie_id", "title", "genres"],
                encoding="unicode_escape",
            ),
        )

        self.raw_ratings = FileObject(
            file_path=self.base_folder / "01_raw_data" / "ratings.dat",
            read_args=ReadArgs(
                sep="::",
                engine="python",
                header=None,
                names=["user_id", "movie_id", "rating", "timestamp"],
            ),
        )

        self.processed_users = FileObject(
            file_path=self.base_folder / "02_processed_data" / "users_processed.parquet"
        )
        self.processed_movies = FileObject(
            file_path=self.base_folder
            / "02_processed_data"
            / "movies_processed.parquet"
        )

        self.processed_ratings = FileObject(
            file_path=self.base_folder
            / "02_processed_data"
            / "processed_ratings.parquet"
        )

        self.train = FileObject(
            file_path=self.base_folder / "03_train_val_test_data" / "train.parquet"
        )
        self.val = FileObject(
            file_path=self.base_folder / "03_train_val_test_data" / "val.parquet"
        )

        self.test = FileObject(
            file_path=self.base_folder / "03_train_val_test_data" / "test.parquet"
        )

        self.inference_model_input = FileObject(
            file_path=self.base_folder
            / "04_inference_model_input_data"
            / "inference.parquet"
        )

        self.inference_model_output = FileObject(
            file_path=self.base_folder
            / "05_inference_model_output_data"
            / "inference.parquet"
        )
