import pandas as pd
import torch
from torchmetrics.retrieval import (
    RetrievalMAP,
    RetrievalNormalizedDCG,
    RetrievalPrecision,
    RetrievalRecall,
)


class RankingEvaluator:
    def __init__(self, list_k, pred_col="pred", label_col="label", qid_col="qid"):
        self.list_k = list_k
        self.pred_col = pred_col
        self.label_col = label_col
        self.qid_col = qid_col
        self.ndcg_metrics_obj = {}
        self.recall_metrics_obj = {}
        self.precision_metrics_obj = {}
        self.map_metrics_obj = {}
        self.recall_metrics = {}
        self.precision_metrics = {}
        self.map_metrics = {}
        self.init()

    def init_metrics(self):
        self.metrics = {
            "k": self.list_k,
            "ndcg@k": [],
            "recall@k": [],
            "precision@k": [],
            "map@k": [],
        }

    def init(self):
        for k in self.list_k:
            self.ndcg_metrics_obj[k] = RetrievalNormalizedDCG(top_k=k)
            self.recall_metrics_obj[k] = RetrievalRecall(top_k=k)
            self.precision_metrics_obj[k] = RetrievalPrecision(top_k=k)
            self.map_metrics_obj[k] = RetrievalMAP(top_k=k)
        return self

    def evaluate(self, data):
        self.init_metrics()

        y_true, y_pred, indexes = (
            data[self.label_col],
            data[self.pred_col],
            data[self.qid_col],
        )
        if isinstance(y_true, pd.Series):
            y_true = y_true.to_numpy()
        if isinstance(y_pred, pd.Series):
            y_pred = y_pred.to_numpy()
        if isinstance(indexes, pd.Series):
            indexes = indexes.to_numpy()
        y_true = torch.tensor(y_true)
        y_pred = torch.tensor(y_pred).float()
        indexes = torch.tensor(indexes)
        for k, ndcg_obj in self.ndcg_metrics_obj.items():
            self.metrics["ndcg@k"].append(ndcg_obj(y_pred, y_true, indexes).item())
        for k, prec_obj in self.precision_metrics_obj.items():
            self.metrics["precision@k"].append(prec_obj(y_pred, y_true, indexes).item())
        for k, recall_obj in self.recall_metrics_obj.items():
            self.metrics["recall@k"].append(recall_obj(y_pred, y_true, indexes).item())
        for k, map_obj in self.map_metrics_obj.items():
            self.metrics["map@k"].append(map_obj(y_pred, y_true, indexes).item())
        return self.metrics
