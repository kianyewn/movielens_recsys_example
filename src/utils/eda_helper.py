import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import entropy


def default_bin_value_counts_categorical_feature(val, feature, top_k=10):
    vc = val[feature].astype(str).value_counts(dropna=False, normalize=True)
    non_null_vc = vc[vc.index != np.nan]
    top_k_vc = non_null_vc.iloc[:top_k]
    top_k_vc = top_k_vc.sort_index()

    if np.nan not in vc:
        null_vc = pd.Series([0], index=["np.nan"])
    else:
        null_vc = vc[vc.index == np.nan]

    non_top_k_vc = pd.Series([non_null_vc.iloc[top_k:].sum()], index=["Others"])

    return (
        pd.concat([top_k_vc, non_top_k_vc, null_vc])
        if np.nan not in vc
        else pd.concat([top_k_vc, non_top_k_vc])
    )


def default_bin_numeric_feature(df, feature, bins=10, retbins=False):
    df_feat_binned, breaks = pd.qcut(
        df[feature], q=bins, retbins=retbins, duplicates="drop"
    )
    breaks = breaks.tolist() + [np.inf]
    if retbins:
        return df_feat_binned, breaks
    return df_feat_binned


def default_bin_categorical_feature(df, feature, bins=10, retbins=False):
    vc = df[feature].astype(str).value_counts()
    non_top_k_cols = vc.index[bins:]
    mapping = {col: "Others" if col in non_top_k_cols else col for col in vc.index}
    df_feat_binned = df[feature].map(mapping).astype(str)

    breaks = df_feat_binned.unique().tolist()
    if retbins:
        return df_feat_binned, breaks
    return df_feat_binned


def calculate_psi(expected, actual, buckettype="bins", buckets=10, axis=0):
    """
    Calculate the Population Stability Index (PSI).

    Parameters:
    -----------
    expected: numpy array of original values
    actual: numpy array of new values
    buckettype: type of strategy for creating buckets, bins splits into even splits, quantiles splits into quantile buckets
    buckets: number of quantiles to use in bucketing variables
    axis: axis by which variables are defined, 0 for vertical, 1 for horizontal

    Returns:
    --------
    psi_value: calculated PSI value
    """

    def psi(expected_array, actual_array, buckets):
        """Calculate PSI for a single variable"""

        # Add a small value to avoid division by zero
        epsilon = 1e-6

        # Ensure both arrays sum to 1
        expected_array = expected_array / expected_array.sum()
        actual_array = actual_array / actual_array.sum()

        # Calculate PSI
        psi_value = sum(
            (expected_array - actual_array)
            * np.log((expected_array + epsilon) / (actual_array + epsilon))
        )

        return psi_value

    # Calculate PSI
    psi_value = psi(expected, actual, buckets)

    return psi_value


def compare_distributions(train_df, test_df, val_df, feature, bins=10, figsize=(14, 8)):
    """
    Compare distributions of a feature between train, test, and validation sets.
    Also calculates PSI between distributions.

    Parameters:
    -----------
    train_df : pandas DataFrame
        Training dataset
    test_df : pandas DataFrame
        Test dataset
    val_df : pandas DataFrame
        Validation dataset
    feature : str
        Feature name to compare
    bins : int
        Number of bins for numerical features
    figsize : tuple
        Figure size

    Returns:
    --------
    fig : matplotlib figure
    psi_values : tuple of PSI values (train_test, train_val, val_test)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])

    # Handle numerical features
    if pd.api.types.is_numeric_dtype(train_df[feature]):
        # Create bins based on training data
        train_feat_binned, train_bins = pd.qcut(
            train_df[feature], q=bins, retbins=True, duplicates="drop"
        )
        train_bins = train_bins.tolist() + [np.inf]
        test_feat_binned = pd.cut(
            test_df[feature], bins=train_bins, include_lowest=True
        )
        val_feat_binned = pd.cut(val_df[feature], bins=train_bins, include_lowest=True)

        # Get value counts
        train_counts = (
            train_feat_binned.astype(str)
            .value_counts(dropna=False, normalize=True)
            .sort_index()
        )
        test_counts = (
            test_feat_binned.astype(str)
            .value_counts(dropna=False, normalize=True)
            .sort_index()
        )
        val_counts = (
            val_feat_binned.astype(str)
            .value_counts(dropna=False, normalize=True)
            .sort_index()
        )
    else:
        # For categorical features
        # train_counts = (
        #     train_df[feature]
        #     .astype(str)
        #     .value_counts(dropna=False, normalize=True)
        #     .sort_index()
        # )
        # test_counts = (
        #     test_df[feature]
        #     .astype(str)
        #     .value_counts(dropna=False, normalize=True)
        #     .sort_index()
        # )
        # val_counts = (
        #     val_df[feature]
        #     .astype(str)
        #     .value_counts(dropna=False, normalize=True)
        #     .sort_index()
        # )

        train_counts = default_bin_value_counts_categorical_feature(
            train_df, feature, top_k=bins
        )
        test_counts = default_bin_value_counts_categorical_feature(
            test_df, feature, top_k=bins
        )
        val_counts = default_bin_value_counts_categorical_feature(
            val_df, feature, top_k=bins
        )
        print("hi")
        # Ensure all have the same categories
        all_cats = sorted(
            set(train_counts.index) | set(test_counts.index) | set(val_counts.index)
        )
        train_counts = train_counts.reindex(all_cats, fill_value=0)
        test_counts = test_counts.reindex(all_cats, fill_value=0)
        val_counts = val_counts.reindex(all_cats, fill_value=0)

    # Set width of bars
    bar_width = 0.25
    index = np.arange(len(train_counts))

    # Create bars
    train_bars = ax1.bar(  # noqa: F841
        index - bar_width,
        train_counts.values,
        bar_width,
        label="Train",
        color="skyblue",
        alpha=0.8,
    )
    val_bars = ax1.bar(  # noqa: F841
        index,
        val_counts.values,
        bar_width,
        label="Validation",
        color="lightgreen",
        alpha=0.8,
    )
    test_bars = ax1.bar(  # noqa: F841
        index + bar_width,
        test_counts.values,
        bar_width,
        label="Test",
        color="lightcoral",
        alpha=0.8,
    )

    # Add labels, title and legend
    ax1.set_xlabel(feature)
    ax1.set_ylabel("Frequency")
    ax1.set_title(f"Distribution of {feature} in Train, Validation, and Test Sets")
    ax1.set_xticks(index)
    ax1.set_xticklabels(train_counts.index, rotation=45, ha="right")
    ax1.legend()

    # Calculate PSI
    psi_train_test = calculate_psi(train_counts.values, test_counts.values)
    psi_train_val = calculate_psi(train_counts.values, val_counts.values)
    psi_val_test = calculate_psi(val_counts.values, test_counts.values)

    # Plot PSI values
    psi_data = [psi_train_test, psi_train_val, psi_val_test]
    psi_labels = ["Train vs Test", "Train vs Val", "Val vs Test"]
    psi_colors = ["lightcoral", "lightgreen", "lightskyblue"]

    ax2.bar(psi_labels, psi_data, color=psi_colors)
    ax2.set_ylabel("PSI Value")
    ax2.set_title("Distribution Difference (Population Stability Index)")

    # Add PSI interpretation
    for i, v in enumerate(psi_data):
        color = "green" if v < 0.1 else ("orange" if v < 0.2 else "red")
        ax2.text(i, v + 0.01, f"{v:.4f}", ha="center", color=color, fontweight="bold")

    # Add PSI interpretation guide
    ax2.axhline(y=0.1, color="green", linestyle="--", alpha=0.5)
    ax2.axhline(y=0.2, color="red", linestyle="--", alpha=0.5)
    ax2.text(2.6, 0.05, "PSI < 0.1: No significant change", color="green")
    ax2.text(2.6, 0.15, "0.1 ≤ PSI < 0.2: Moderate change", color="orange")
    ax2.text(2.6, 0.25, "PSI ≥ 0.2: Significant change", color="red")

    # Adjust layout
    fig.tight_layout()

    return fig, (psi_train_test, psi_train_val, psi_val_test)


def compare_distributions_kl_divergence(
    train_df, test_df, val_df, feature, bins=10, figsize=(14, 8)
):
    """
    Compare distributions of a feature between train, test, and validation sets.
    Also calculates KL divergence between distributions.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])

    # Handle numerical features
    if pd.api.types.is_numeric_dtype(train_df[feature]):
        # Create bins based on training data
        train_feat_binned, train_bins = pd.qcut(
            train_df[feature], q=bins, retbins=True, duplicates="drop"
        )
        train_bins = train_bins.tolist() + [np.inf]
        test_feat_binned = pd.cut(
            test_df[feature], bins=train_bins, include_lowest=True
        )
        val_feat_binned = pd.cut(val_df[feature], bins=train_bins, include_lowest=True)

        # Get value counts
        train_counts = (
            train_feat_binned.astype(str)
            .value_counts(dropna=False, normalize=True)
            .sort_index()
        )
        test_counts = (
            test_feat_binned.astype(str)
            .value_counts(dropna=False, normalize=True)
            .sort_index()
        )
        val_counts = (
            val_feat_binned.astype(str)
            .value_counts(dropna=False, normalize=True)
            .sort_index()
        )
    else:
        # For categorical features
        train_counts = (
            train_df[feature]
            .astype(str)
            .value_counts(dropna=False, normalize=True)
            .sort_index()
        )
        test_counts = (
            test_df[feature]
            .astype(str)
            .value_counts(dropna=False, normalize=True)
            .sort_index()
        )
        val_counts = (
            val_df[feature]
            .astype(str)
            .value_counts(dropna=False, normalize=True)
            .sort_index()
        )

        # Ensure all have the same categories
        all_cats = sorted(
            set(train_counts.index) | set(test_counts.index) | set(val_counts.index)
        )
        train_counts = train_counts.reindex(all_cats, fill_value=0)
        test_counts = test_counts.reindex(all_cats, fill_value=0)
        val_counts = val_counts.reindex(all_cats, fill_value=0)

    # Set width of bars
    bar_width = 0.25
    index = np.arange(len(train_counts))

    # Create bars
    train_bars = ax1.bar(  # noqa: F841
        index - bar_width,
        train_counts.values,
        bar_width,
        label="Train",
        color="skyblue",
        alpha=0.8,
    )
    val_bars = ax1.bar(  # noqa: F841
        index,
        val_counts.values,
        bar_width,
        label="Validation",
        color="lightgreen",
        alpha=0.8,
    )
    test_bars = ax1.bar(  # noqa: F841
        index + bar_width,
        test_counts.values,
        bar_width,
        label="Test",
        color="lightcoral",
        alpha=0.8,
    )

    # Add labels, title and legend
    ax1.set_xlabel(feature)
    ax1.set_ylabel("Frequency")
    ax1.set_title(f"Distribution of {feature} in Train, Validation, and Test Sets")
    ax1.set_xticks(index)
    ax1.set_xticklabels(train_counts.index, rotation=45, ha="right")
    ax1.legend()

    # Calculate KL divergence
    # Add small epsilon to avoid division by zero
    epsilon = 1e-10
    train_probs = train_counts.values + epsilon
    train_probs = train_probs / train_probs.sum()

    test_probs = test_counts.values + epsilon
    test_probs = test_probs / test_probs.sum()

    val_probs = val_counts.values + epsilon
    val_probs = val_probs / val_probs.sum()

    kl_train_test = entropy(train_probs, test_probs)
    kl_train_val = entropy(train_probs, val_probs)
    kl_val_test = entropy(val_probs, test_probs)

    # Plot KL divergence
    kl_data = [kl_train_test, kl_train_val, kl_val_test]
    kl_labels = ["Train vs Test", "Train vs Val", "Val vs Test"]
    kl_colors = ["lightcoral", "lightgreen", "lightskyblue"]

    ax2.bar(kl_labels, kl_data, color=kl_colors)
    ax2.set_ylabel("KL Divergence")
    ax2.set_title("Distribution Difference (KL Divergence)")

    for i, v in enumerate(kl_data):
        ax2.text(i, v + 0.01, f"{v:.4f}", ha="center")

    # Adjust layout
    fig.tight_layout()

    return fig, (kl_train_test, kl_train_val, kl_val_test)


def analyze_numeric_feature(df, feature, bins=10, figsize=(15, 6)):
    """
    Analyze a numerical feature with histogram (showing percentages) and boxplot.
    """
    # Create subplots with gridspec to control relative widths
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 2, width_ratios=[2, 1], wspace=0.1)

    # Histogram subplot
    ax1 = fig.add_subplot(gs[0])

    # Get binned data using default_bin_numeric_feature
    binned_data, bin_edges = default_bin_numeric_feature(
        df, feature, bins=bins, retbins=True
    )
    value_counts = binned_data.value_counts(
        dropna=False
    ).sort_index()  # Get actual counts
    percentages = value_counts / len(binned_data)  # Calculate percentages separately

    # Create bars for the histogram
    bar_positions = range(len(value_counts))
    bars = ax1.bar(
        bar_positions,
        value_counts.values,  # Use counts instead of percentages
        edgecolor="black",
        alpha=0.7,
        width=0.8,
    )

    # Add percentage labels on top of each bar
    for i, (count, percentage) in enumerate(zip(bars, percentages)):
        height = count.get_height()
        ax1.text(
            count.get_x() + count.get_width() / 2,
            height,
            f"{percentage*100:.1f}%",
            ha="center",
            va="bottom",
        )

    # Set x-ticks with interval labels
    ax1.set_xticks(bar_positions)
    ax1.set_xticklabels(
        [str(interval) for interval in value_counts.index], rotation=45, ha="right"
    )

    ax1.set_title(f"Distribution of {feature}")
    ax1.set_xlabel(feature)
    ax1.set_ylabel("Count")

    # Boxplot subplot
    ax2 = fig.add_subplot(gs[1])

    # Calculate statistics once (handling NaN values properly)
    non_null_data = df[feature].dropna()
    stats = {
        "Count": len(non_null_data),
        "Missing": len(df[feature]) - len(non_null_data),
        "Mean": non_null_data.mean(),
        "Std": non_null_data.std(),
        "Min": non_null_data.min(),
        "Max": non_null_data.max(),
        "Q1": non_null_data.quantile(0.25),
        "Q2": non_null_data.quantile(0.50),
        "Q3": non_null_data.quantile(0.75),
    }

    # Create boxplot with smaller size and orange color
    bp = ax2.boxplot(
        non_null_data,
        vert=True,
        widths=0.3,
        patch_artist=True,
        boxprops=dict(facecolor="orange", color="black", alpha=0.6),
        medianprops=dict(color="darkred"),
        flierprops=dict(marker="o", markerfacecolor="gray", markersize=4),
    )
    ax2.set_title(f"Boxplot of {feature}")
    ax2.set_xlabel(f"{feature}")

    # Add descriptive statistics as text inside the plot
    stats_text = (
        f"N = {stats['Count']:,}\n"
        f"({stats['Missing']:,} missing)\n"
        f"Mean = {stats['Mean']:.2f}\n"
        f"Std = {stats['Std']:.2f}\n"
        f"Min = {stats['Min']:.2f}\n"
        f"25% = {stats['Q1']:.2f}\n"
        f"50% = {stats['Q2']:.2f}\n"
        f"75% = {stats['Q3']:.2f}\n"
        f"Max = {stats['Max']:.2f}"
    )

    # Position the text in the top right corner of the plot
    ax2.text(
        0.95,  # x position in axes coordinates
        0.95,  # y position in axes coordinates
        stats_text,
        transform=ax2.transAxes,  # Use axes coordinates
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        verticalalignment="top",
        horizontalalignment="right",
    )
    y_range = stats["Max"] - stats["Min"]

    # Adjust boxplot y-axis limits to match the data range
    ax2.set_ylim(
        stats["Min"] - 0.1 * y_range,  # Reduced bottom padding
        stats["Max"] + 0.1 * y_range,
    )

    plt.tight_layout()
    return fig


def analyze_numeric_feature_with_target(df, feature, target, bins=10, figsize=(15, 6)):
    """
    Analyze a numerical feature with histogram and target distribution.

    Parameters:
    -----------
    df : pandas DataFrame
        Input dataframe
    feature : str
        Name of the numerical feature to analyze
    target : str
        Name of the target column
    bins : int, default=10
        Number of bins for the histogram
    figsize : tuple, default=(15, 6)
        Figure size as (width, height)
    """
    fig = plt.figure(figsize=figsize)
    # Increase spacing between subplots to accommodate y-axis labels
    gs = fig.add_gridspec(1, 2, width_ratios=[2, 1], wspace=0.25)

    # Histogram subplot with target average
    ax1 = fig.add_subplot(gs[0])
    ax1_twin = ax1.twinx()  # Create secondary y-axis

    # Get binned data and calculate target means per bin
    binned_data, bin_edges = default_bin_numeric_feature(
        df, feature, bins=bins, retbins=True
    )
    value_counts = binned_data.value_counts(dropna=False).sort_index()
    percentages = value_counts / len(binned_data)

    # Calculate mean target value for each bin
    target_means = df.groupby(binned_data, dropna=False, observed=True)[target].mean()
    target_means = target_means.reindex(value_counts.index)

    # Create bars for the histogram
    bar_positions = range(len(value_counts))
    bars = ax1.bar(
        bar_positions, value_counts.values, edgecolor="black", alpha=0.7, width=0.8
    )

    # Plot target means line
    line = ax1_twin.plot(
        bar_positions,
        target_means.values,
        color="goldenrod",
        linewidth=2,
        marker="o",
        label=f"Mean {target}",
    )

    # Add percentage labels on top of each bar
    for i, (count, percentage) in enumerate(zip(bars, percentages)):
        height = count.get_height()
        ax1.text(
            count.get_x() + count.get_width() / 2,
            height,
            f"{percentage*100:.1f}%",
            ha="center",
            va="bottom",
        )

    # Set x-ticks with interval labels
    ax1.set_xticks(bar_positions)
    ax1.set_xticklabels(
        [str(interval) for interval in value_counts.index], rotation=45, ha="right"
    )

    ax1.set_title(f"Distribution of `{feature}` with `{target}`")
    ax1.set_xlabel(feature)
    ax1.set_ylabel("Count")
    ax1_twin.set_ylabel(f"Mean `{target}`", color="goldenrod")
    ax1_twin.tick_params(axis="y", labelcolor="goldenrod")
    # Remove grid lines for secondary axes
    ax1_twin.grid(False)

    # Add legend for target mean line
    ax1_twin.legend(loc="upper right")

    # Boxplot subplot
    ax2 = fig.add_subplot(gs[1])

    # Calculate statistics
    non_null_data = df[feature].dropna()
    non_null_target = df[target][non_null_data.index]

    stats = {
        "Count": len(non_null_data),
        "Missing": len(df[feature]) - len(non_null_data),
        "Mean": non_null_data.mean(),
        "Std": non_null_data.std(),
        "Min": non_null_data.min(),
        "Max": non_null_data.max(),
        "Q1": non_null_data.quantile(0.25),
        "Q2": non_null_data.quantile(0.50),
        "Q3": non_null_data.quantile(0.75),
        "Target_Mean": non_null_target.mean(),
        "Target_Std": non_null_target.std(),
    }

    # Create boxplot with position adjustment
    bp = ax2.boxplot(
        non_null_data,
        vert=True,
        widths=0.3,
        positions=[0.5],  # Center the boxplot
        patch_artist=True,
        boxprops=dict(facecolor="orange", color="black", alpha=0.6),
        medianprops=dict(color="darkred"),
        flierprops=dict(marker="o", markerfacecolor="gray", markersize=4),
    )
    ax2.set_title(f"Boxplot of `{feature}`")
    ax2.set_xlabel(f"{feature}")

    # Remove x-ticks since we only have one boxplot
    ax2.set_xticks([])

    # Add descriptive statistics with target info
    stats_text = (
        f"N = {stats['Count']:,}\n"
        f"({stats['Missing']:,} missing)\n"
        f"Mean = {stats['Mean']:.2f}\n"
        f"Std = {stats['Std']:.2f}\n"
        f"Min = {stats['Min']:.2f}\n"
        f"25% = {stats['Q1']:.2f}\n"
        f"50% = {stats['Q2']:.2f}\n"
        f"75% = {stats['Q3']:.2f}\n"
        f"Max = {stats['Max']:.2f}\n"
        f"\n{target}:\n"
        f"Mean = {stats['Target_Mean']:.2f}\n"
        f"Std = {stats['Target_Std']:.2f}"
    )

    # Position the text in the top right corner
    ax2.text(
        1.05,
        0.95,
        stats_text,
        transform=ax2.transAxes,
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        verticalalignment="top",
        horizontalalignment="right",
    )

    y_range = stats["Max"] - stats["Min"]
    ax2.set_ylim(stats["Min"] - 0.1 * y_range, stats["Max"] + 0.1 * y_range)

    # plt.tight_layout()
    return fig


def analyze_numeric_feature_with_targets(df, feature, targets, bins=10, figsize=(15, 6)):
    """
    Analyze a numerical feature with histogram and multiple target distributions.
    Each target gets its own y-axis to handle different scales.
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 2, width_ratios=[2, 1], wspace=0.25)

    # Histogram subplot with target averages
    ax1 = fig.add_subplot(gs[0])
    twin_axes = [ax1.twinx()]  # Create first secondary y-axis
    if len(targets) > 1:
        for i in range(len(targets) - 1):   
            twin_axes.append(ax1.twinx())  # Create second secondary y-axis if needed
            # Offset the second axis further right
            # twin_axes[i+1].spines['right'].set_position(('outward', 60))
            # Move the ticks and labels to the right for better separation
            twin_axes[i+1].tick_params(axis='y', pad=(i+1) * 30)  # Add padding between ticks and labels
            # print('hello') 
    # Get binned data and calculate target means per bin
    binned_data, bin_edges = default_bin_numeric_feature(df, feature, bins=bins, retbins=True)
    value_counts = binned_data.value_counts(dropna=False).sort_index()
    percentages = value_counts / len(binned_data)

    # Create bars for the histogram
    bar_positions = range(len(value_counts))
    bars = ax1.bar(
        bar_positions, value_counts.values, edgecolor="black", alpha=0.7, width=0.8
    )

    # Colors for different targets
    # Colors and styles for different targets
    colors = [
        'goldenrod',      # Warm yellow/gold - keep as requested
        'C5',
        '#2ca02c',       # Warm green that matches well with goldenrod
        '#2ecc71',      # Emerald green
        '#2E8B57',       # Sea green - softer than forestgreen
        '#E9967A',    # Dark salmon
        '#00B294',    # Teal
        '#4682B4',       # Steel blue - more professional than plain blue
        '#CD5C5C',       # Indian red - softer than brown
        '#4B0082',       # Indigo - more distinctive than purple
    ]
    styles = ['-', '--', ':', '-.', '-']
    lines = []

    # Plot target means lines
    for idx, (target, twin_ax) in enumerate(zip(targets, twin_axes)):
        # Calculate mean target value for each bin
        target_means = df.groupby(binned_data, dropna=False, observed=True)[target].mean()
        target_means = target_means.reindex(value_counts.index)

        # print('hello')

        # Plot line
        line = twin_ax.plot(
            bar_positions,
            target_means.values,
            color=colors[idx],
            linewidth=2,
            marker="o" if idx == 0 else None,
            label=f"Mean {target}",
            linestyle=styles[idx % len(styles)],
        )[0]
        lines.append(line)

        # Set axis formatting
        # twin_ax.set_ylabel(f"Mean_{target}", color=colors[idx])
        twin_ax.tick_params(axis="y", labelcolor=colors[idx])
        # Adjust tick label format to be more compact
        twin_ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.2f}"))
        twin_ax.grid(False)
    twin_axes[-1].set_ylabel("Mean", labelpad=0, color=colors[0]) 
    # Add percentage labels on top of each bar
    for i, (count, percentage) in enumerate(zip(bars, percentages)):
        height = count.get_height()
        ax1.text(
            count.get_x() + count.get_width() / 2,
            height,
            f"{percentage*100:.1f}%",
            ha="center",
            va="bottom",
        )

    # Set x-ticks with interval labels
    ax1.set_xticks(bar_positions)
    ax1.set_xticklabels(
        [str(interval) for interval in value_counts.index], rotation=45, ha="right"
    )

    ax1.set_title(f"Distribution of `{feature}` with Target Values")
    ax1.set_xlabel(feature)
    ax1.set_ylabel("Count")

    # Add legend for target mean lines
    ax1.legend(lines, [f"Mean {target}" for target in targets], loc="upper right", bbox_to_anchor=(1.12, 1.12))

    # Boxplot subplot
    ax2 = fig.add_subplot(gs[1])

    # Calculate statistics
    non_null_data = df[feature].dropna()
    stats = {
        "Count": len(non_null_data),
        "Missing": len(df[feature]) - len(non_null_data),
        "Mean": non_null_data.mean(),
        "Std": non_null_data.std(),
        "Min": non_null_data.min(),
        "Max": non_null_data.max(),
        "Q1": non_null_data.quantile(0.25),
        "Q2": non_null_data.quantile(0.50),
        "Q3": non_null_data.quantile(0.75),
    }

    # Add target statistics
    for target in targets:
        non_null_target = df[target][non_null_data.index]
        stats[f"{target}_Mean"] = non_null_target.mean()
        stats[f"{target}_Std"] = non_null_target.std()

    # Add descriptive statistics
    stats_text = (
        f"N = {stats['Count']:,}\n"
        f"({stats['Missing']:,} missing)\n"
        f"Mean = {stats['Mean']:.2f}\n"
        f"Std = {stats['Std']:.2f}\n"
        f"Min = {stats['Min']:.2f}\n"
        f"25% = {stats['Q1']:.2f}\n"
        f"50% = {stats['Q2']:.2f}\n"
        f"75% = {stats['Q3']:.2f}\n"
        f"Max = {stats['Max']:.2f}\n"
    )

    # Add target statistics to text
    for target in targets:
        stats_text += f"\n{target}:\n"
        stats_text += f"Mean = {stats[f'{target}_Mean']:.2f}\n"
        stats_text += f"Std = {stats[f'{target}_Std']:.2f}\n"

    # Create boxplot
    bp = ax2.boxplot(
        non_null_data,
        vert=True,
        widths=0.3,
        positions=[0.5],
        patch_artist=True,
        boxprops=dict(facecolor="orange", color="black", alpha=0.6),
        medianprops=dict(color="darkred"),
        flierprops=dict(marker="o", markerfacecolor="gray", markersize=4),
    )
    ax2.set_title(f"Boxplot of {feature}")
    ax2.set_xlabel(feature)
    ax2.set_xticks([])

    ax2.text(
        1.05,
        0.95,
        stats_text,
        transform=ax2.transAxes,
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        verticalalignment="top",
        horizontalalignment="right",
    )

    y_range = stats["Max"] - stats["Min"]
    ax2.set_ylim(stats["Min"] - 0.1 * y_range, stats["Max"] + 0.1 * y_range)

    plt.subplots_adjust(wspace=0.3)  # Increase space between subplots
    return fig


def analyze_categorical_feature_with_targets(df, feature, targets, bins=10, figsize=(15, 6)):
    """
    Analyze a categorical feature with bar plot and multiple target distributions.
    Each target gets its own y-axis to handle different scales.
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 2, width_ratios=[6, 1], wspace=0.25)

    # Bar plot subplot with target averages
    ax1 = fig.add_subplot(gs[0])
    twin_axes = [ax1.twinx()]  # Create first secondary y-axis
    if len(targets) > 1:
        for i in range(len(targets) - 1):   
            twin_axes.append(ax1.twinx())  # Create second secondary y-axis if needed
            twin_axes[i+1].tick_params(axis='y', pad=(i+1) * 30)  # Add padding between ticks and labels

    # Get binned data and value counts
    binned_data = default_bin_categorical_feature(df, feature, bins=bins)
    value_counts = binned_data.value_counts(dropna=False).sort_values(ascending=False)
    value_counts = value_counts.sort_index()
    percentages = value_counts / len(binned_data)

    # Create bars for the histogram
    bar_positions = range(len(value_counts))
    bars = ax1.bar(
        bar_positions, value_counts.values, edgecolor="black", alpha=0.7, width=0.8
    )

    # Colors and styles for different targets
    colors = [
        'goldenrod',      # Warm yellow/gold
        'C5',            # Matplotlib cycle color 5
        '#2ca02c',       # Warm green
        '#2ecc71',       # Emerald green
        '#2E8B57',       # Sea green
        '#E9967A',       # Dark salmon
        '#00B294',       # Teal
        '#4682B4',       # Steel blue
    ]
    styles = ['-', '--', ':', '-.', '-']
    lines = []

    # Plot target means lines
    for idx, (target, twin_ax) in enumerate(zip(targets, twin_axes)):
        # Calculate mean target value for each category
        target_means = df.groupby(binned_data, dropna=False, observed=True)[target].mean()
        target_means = target_means.reindex(value_counts.index)

        # Plot line
        line = twin_ax.plot(
            bar_positions,
            target_means.values,
            color=colors[idx],
            linewidth=2,
            marker="o" if idx == 0 else None,
            label=f"Mean {target}",
            linestyle=styles[idx % len(styles)],
        )[0]
        lines.append(line)

        # Set axis formatting
        twin_ax.tick_params(axis="y", labelcolor=colors[idx])
        twin_ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.2f}"))
        twin_ax.grid(False)
    
    twin_axes[-1].set_ylabel("Mean", labelpad=0, color=colors[0])

    # Add percentage labels on top of each bar
    for i, (count, percentage) in enumerate(zip(bars, percentages)):
        height = count.get_height()
        ax1.text(
            count.get_x() + count.get_width() / 2,
            height,
            f"{percentage*100:.1f}%",
            ha="center",
            va="bottom",
        )

    # Set x-ticks with category labels
    ax1.set_xticks(bar_positions)
    ax1.set_xticklabels(value_counts.index, rotation=45, ha="right")

    ax1.set_title(f"Distribution of `{feature}` with Target Values")
    ax1.set_xlabel(feature)
    ax1.set_ylabel("Count")

    # Add legend for target mean lines
    ax1.legend(lines, [f"Mean {target}" for target in targets], loc="upper right", bbox_to_anchor=(1.12, 1.12))

    # Statistics subplot
    ax2 = fig.add_subplot(gs[1])

    # Calculate statistics
    stats = {
        "Categories": df[feature].nunique(dropna=False),
        "Total": len(df[feature]),
        "Missing": df[feature].isna().sum(),
    }

    # Add target statistics
    for target in targets:
        non_null_data = df[~df[feature].isna()]
        non_null_target = non_null_data[target]
        stats[f"{target}_Mean"] = non_null_target.mean()
        stats[f"{target}_Std"] = non_null_target.std()

    # Add descriptive statistics
    stats_text = (
        f"Categories: {stats['Categories']}\n"
        f"Total: {stats['Total']:,}\n"
        f"Missing: {stats['Missing']:,}\n"
    )

        # Add top 3 categories
    stats_text += "\nTop 3 Categories:\n"
    top_vcs = value_counts.sort_values(ascending=False).head(3)
    for cat, count in top_vcs.items():
        stats_text += f"{cat}: {count:,} ({count/len(df)*100:.1f}%)\n"
        
    # Add target statistics to text
    for target in targets:
        stats_text += f"\n{target}:\n"
        stats_text += f"Mean: {stats[f'{target}_Mean']:.2f}\n"
        stats_text += f"Std: {stats[f'{target}_Std']:.2f}"



    ax2.text(
        0.5,
        1,
        stats_text,
        transform=ax2.transAxes,
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="whitesmoke", alpha=0.8, linewidth=0.5, edgecolor='darkgrey'),
        verticalalignment="top",
        horizontalalignment="center",
    )
    ax2.axis('off')

    plt.subplots_adjust(wspace=0.3)  # Increase space between subplots
    return fig
