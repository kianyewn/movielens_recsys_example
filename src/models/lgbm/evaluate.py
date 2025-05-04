import pandas as pd

from configs.lgbm_config import LGBMConfig
from src.evaluation.ranking_evaluator import RankingEvaluator
from src.models.lgbm.inference import InferenceEngine


class EvaluateEngine:
    def __init__(self, lgbm_config: LGBMConfig):
        self.lgbm_config = lgbm_config
        self.inference_engine = InferenceEngine(lgbm_config)
        self.evaluator = RankingEvaluator(list_k=[1, 2, 3, 4, 5])

    def evaluate(self, test):
        test["pred"] = self.inference_engine.predict(test)
        self.evaluator.evaluate(test)
        self.best_model_results = pd.DataFrame(self.evaluator.metrics)

        test["pred"] = self.inference_engine.predict_default(test)
        self.evaluator.evaluate(test)
        self.default_model_results = pd.DataFrame(self.evaluator.metrics)
        self.format_results()
        return self

    def format_results(self):
        self.best_model_results = self.best_model_results.set_index("k").add_prefix(
            "best_lgbm_"
        )
        self.default_model_results = self.default_model_results.set_index(
            "k"
        ).add_prefix("default_lgbm_")
        self.eval_results = pd.concat(
            [self.best_model_results, self.default_model_results], axis=1
        )
        sorted_columns = sorted(
            self.eval_results.columns, key=lambda x: x.split("_", 2)[-1]
        )
        self.eval_results = self.eval_results[sorted_columns].reset_index()
        return self

    def save_results(self):
        self.eval_results.to_csv(self.lgbm_config.results_path, index=False)
        return self
