import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.eda_helper import (
    analyze_categorical_feature,
    analyze_categorical_feature_with_target,
    analyze_categorical_feature_with_targets,
    analyze_numeric_feature_with_target,
    analyze_numeric_feature_with_targets,
    compare_distributions,
    compare_distributions_kl_divergence,
    default_bin_categorical_feature,
    default_bin_numeric_feature,
    eda_categorical,
    eda_label,
    eda_numerical,
    get_all_stats,
    get_time_stats,
    value_counts,
)


def test_value_counts(users_df):
    data = value_counts(users_df, "gender")
    assert data.to_dict() == {
        "var": {0: "gender", 1: "gender", 2: "gender", 3: "gender"},
        "value": {0: "M", 1: "F", 2: "missing", 3: "total"},
        "count": {0: 4331, 1: 1709, 2: 0, 3: 6040},
        "proportion": {0: 0.7170529801324503, 1: 0.28294701986754967, 2: 0.0, 3: 1.0},
    }


def test_analyze_categorical_feature(users_df):
    fig = analyze_categorical_feature(users_df, "gender")
    assert isinstance(fig, plt.Figure)


def test_get_time_stats(ratings_df):
    ratings_samp = ratings_df.head(10000).copy()
    ratings_samp["date"] = pd.to_datetime(ratings_samp["timestamp"], unit="s")
    ratings_samp["year_month"] = ratings_samp["date"].dt.strftime("%m")
    dat = get_time_stats(
        ratings_samp[ratings_samp["rating"] == 5], "rating", "year_month"
    )
    assert dat.columns.tolist() == ["year_month", "rating", "count", "proportion"]


def test_get_all_stats(users_df):
    np.random.seed(42)
    users_df["target"] = np.random.randint(0, 2, size=len(users_df))
    dat = get_all_stats(users_df, "gender", "target")
    assert dat.to_dict() == {
        "gender": {0: "F", 1: "M"},
        "count": {0: 1709, 1: 4331},
        "proportion": {0: 0.28294701986754967, 1: 0.7170529801324503},
        "target": {0: 0.497366881217086, 1: 0.4964211498499192},
    }


def test_plotting_utils(users_df, test):
    users = users_df.copy()
    users["year_month"] = np.random.choice(
        ["01", "02", "03", "04", "05"], size=len(users)
    )
    users["target"] = np.random.choice([0, 1], size=len(users))
    users["age_binned"] = default_bin_numeric_feature(users, "age")
    users["zip_binned"] = default_bin_categorical_feature(users, "zip")
    _ = analyze_numeric_feature_with_target(users, "age", target="target")
    _ = analyze_numeric_feature_with_targets(users, "age", targets=["target", "target"])
    _ = analyze_categorical_feature_with_targets(
        users, "gender", targets=["target", "target"]
    )
    _ = analyze_categorical_feature_with_target(users, "gender", target="target")
    _ = analyze_categorical_feature(users, "gender", bins=10)
    _ = eda_numerical(users, "age", label="target", timecol="year_month")
    _ = eda_categorical(users, "gender", label="target", timecol="year_month")
    _ = eda_label(users, "target", "year_month")

    assert True


def test_compare_distributions(train, val, test):
    fig, (psi_train_test, psi_train_val, psi_val_test) = compare_distributions(
        train, val, test, "age"
    )
    assert isinstance(psi_train_test, float)
    assert isinstance(psi_train_val, float)
    assert isinstance(psi_val_test, float)


def test_compare_distributions_kl_divergence(train, val, test):
    fig, (kl_train_test, kl_train_val, kl_val_test) = (
        compare_distributions_kl_divergence(train, val, test, "age")
    )
    assert isinstance(kl_train_test, float)
    assert isinstance(kl_train_val, float)
    assert isinstance(kl_val_test, float)
