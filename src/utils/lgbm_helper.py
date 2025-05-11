import pandas as pd


def get_lgbm_feature_importance_df(model, importance_types=["split", "gain"]):
    feature_importane_dfs = []
    for importance_type in importance_types:
        feature_importance_dict = {"importance_type": importance_type}
        feature_importance_dict["feature_importance"] = (
            model.booster_.feature_importance(importance_type=importance_type)
        )
        feature_importance_dict["feature"] = model.feature_name_
        feature_importance_df = pd.DataFrame(feature_importance_dict).sort_values(
            by="feature_importance", ascending=False
        )
        feature_importance_df["rank"] = feature_importance_df[
            "feature_importance"
        ].rank(method="first", ascending=False)
        feature_importane_dfs.append(feature_importance_df)
    feature_importance_df = pd.concat(feature_importane_dfs, axis=0)
    return feature_importance_df
