import functools
import inspect
import time
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from loguru import logger


def plot_learning_curve(
    xgb_model,
    metrics_to_plot: Optional[List[str]] = None,
    legend_labels: List[str] = ["train", "val"],
):
    """Plots the learning curve for an XGBoost model showing training and validation metrics.

    Args:
        xgb_model (XGBClassifier): A trained XGBoost classifier with evaluation results.
        metrics_to_plot (List[str], optional): List of metrics to plot. Defaults to ["mlogloss"].
        legend_labels (List[str], optional): Labels for the training and validation curves.
            Defaults to ["train", "val"].

    Raises:
        AttributeError: If eval_set was not specified during model training.

    Example:
        >>> model = XGBClassifier()
        >>> model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)])
        >>> plot_learning_curve(model, metrics_to_plot=["mlogloss", "merror"])
    """
    if not xgb_model.evals_result_:
        raise AttributeError("You did not specify `eval_set` in .fit()")
    if isinstance(metrics_to_plot, str):
        metrics_measured = list(list(xgb_model.evals_result_.values())[0].keys())
        metrics_to_plot = [
            metric for metric in metrics_measured if metrics_to_plot in metric
        ]

    evals_result = xgb_model.evals_result_
    for idx, (eval_dtype, result) in enumerate(evals_result.items()):
        for metric in result:
            if metric in metrics_to_plot:
                metric_history = result[metric]
                x = np.arange(len(metric_history))
                plt.plot(x, metric_history, label=f"{legend_labels[idx]}_{metric}")
    plt.legend(loc="right")
    plt.show()
    return


def plot_learning_curve_plotly(
    xgb_model,
    metrics_to_plot: List[str] = ["mlogloss"],
    legend_labels: List[str] = ["train", "val"],
    title="Learning Curve",
):
    """Plots the learning curve for an XGBoost model showing training and validation metrics.

    Args:
        xgb_model (XGBClassifier): A trained XGBoost classifier with evaluation results.
        metrics_to_plot (List[str], optional): List of metrics to plot. Defaults to ["mlogloss"].
        legend_labels (List[str], optional): Labels for the training and validation curves.
            Defaults to ["train", "val"].

    Raises:
        AttributeError: If eval_set was not specified during model training.

    Example:
        >>> model = XGBClassifier()
        >>> model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)])
        >>> plot_learning_curve(model, metrics_to_plot=["mlogloss", "merror"])
    """
    if not xgb_model.evals_result_:
        raise AttributeError("You did not specify `eval_set` in .fit()")
    if isinstance(metrics_to_plot, str):
        metrics_measured = list(list(xgb_model.evals_result_.values())[0].keys())
        metrics_to_plot = [
            metric for metric in metrics_measured if metrics_to_plot in metric
        ]
    evals_result = xgb_model.evals_result_
    fig = go.Figure()
    for idx, (eval_dtype, result) in enumerate(evals_result.items()):
        for metric in result:
            if metric in metrics_to_plot:
                metric_history = result[metric]
                x = np.arange(len(metric_history))
                label = f"{legend_labels[idx]}_{metric}"
                fig.add_trace(
                    go.Scatter(x=x, y=metric_history, mode="lines", name=label)
                )

    fig.update_layout(
        xaxis=dict(title="Iterations"),
        # yaxis=dict(automargin=True),
        yaxis1=dict(
            title="Metric Value",
            color="#1f77b4",  # Extend the y-axis
        ),
        title=dict(
            text=f"<b>{title}</b>",
            x=0.5,
            y=0.9,
            xanchor="center",
        ),
        hovermode="x unified",
    )
    fig.show()
    return


def value_counts(df, column):
    val_count = df[column].value_counts(dropna=False).rename("count")
    val_perc = (
        df[column].value_counts(dropna=False, normalize=True).rename("percentage")
    )
    margin = [df.shape[0], 1]
    vc_table = pd.concat([val_count, val_perc], axis=1)  # .concat(margin, axis=0)
    vc_table.loc["total"] = margin
    return vc_table


def timeit(func):
    """Simple decorator to measure function execution time"""
    # Get the original function's metadata
    func_name = func.__name__
    func_module = func.__module__

    # Get the source file and line number if possible
    try:
        func_file = inspect.getfile(func)
        func_line = inspect.getsourcelines(func)[1]
    except (TypeError, OSError):
        func_file = "<unknown>"
        func_line = 0

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        # Use logger.opt() with a custom record patcher
        logger.opt(depth=1).patch(
            lambda record: record.update(
                function=func_name,
                module=func_module,
                file=func_file,
                line=func_line,
                name=func_module,  # This is what appears before the colon in logs
            )
        ).info(f"Function '{func_name}' executed in {elapsed_time:.4f} seconds")

        return result

    return wrapper
