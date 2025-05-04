import pandas as pd

from configs.data_config import DataConfig
from configs.lgbm_config import LGBMConfig
from src.models.lgbm.evaluate import EvaluateEngine

if __name__ == "__main__":
    # load config
    lgbm_config = LGBMConfig()
    data_config = DataConfig()

    # load data
    test = pd.read_parquet(data_config.test.file_path)

    # evaluate model
    eval_engine = EvaluateEngine(lgbm_config)
    eval_engine.evaluate(test)

    # save results
    eval_engine.save_results()
