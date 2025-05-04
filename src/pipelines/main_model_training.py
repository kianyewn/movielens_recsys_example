import pandas as pd
from torchmetrics.retrieval import RetrievalNormalizedDCG

from configs.data_config import DataConfig
from configs.lgbm_config import LGBMConfig
from src.models.lgbm.train import TrainingEngine

if __name__ == "__main__":
    # load config
    data_config = DataConfig()
    lgbm_config = LGBMConfig()

    # load data
    train = pd.read_parquet(data_config.train.file_path)
    val = pd.read_parquet(data_config.val.file_path)
    test = pd.read_parquet(data_config.test.file_path)

    # train model
    trainer = TrainingEngine(
        train,
        val,
        lgbm_config,
        scorer=RetrievalNormalizedDCG(top_k=5),
        scorer_name="ndcg_at_5",
    )

    # hyperparameter tuning
    trainer.hyperparameter_tuning()
    trainer.best_model_results

    # train model
    trainer.train_default_model()
    trainer.train_best_model()

    # save model
    trainer.save_best_model()
    trainer.save_default_model()
