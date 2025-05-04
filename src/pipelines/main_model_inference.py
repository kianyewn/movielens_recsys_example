from configs.data_config import InferenceDataConfig
from configs.lgbm_config import LGBMConfig
import pandas as pd
from src.models.lgbm.inference import InferenceEngine


if __name__ == "__main__":
    # load config
    data_config = InferenceDataConfig()
    model_config = LGBMConfig()

    # load data
    inference_data = pd.read_parquet(data_config.inference_model_input.file_path)
    inference_data.shape

    # Load Model
    inference_engine = InferenceEngine(model_config)
    inference_engine.score(inference_data)

    # save data
    inference_engine.inference_data.to_parquet(
        data_config.inference_model_output.file_path
    )
