# analysis_utils.py

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

def compute_median_vectors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group by source_file, compute median for each column.
    Returns a DataFrame indexed by source_file.
    """
    return df.groupby("source_file").median()

def plot_median_vectors_line(medians_df, figsize=(10,6), alpha=0.3):
    """
    Line plot of each file's 13-dimensional median vector.
    """
    plt.figure(figsize=figsize)
    x = range(len(medians_df.columns))
    for _, row in medians_df.iterrows():
        plt.plot(x, row.values, alpha=alpha)
    plt.xticks(x, medians_df.columns, rotation=90)
    plt.xlabel("Feature")
    plt.ylabel("Median Value")
    plt.title("Per-file median vectors (line plot)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_median_heatmap(medians_df, figsize=(12,10), cmap="viridis"):
    """
    Heatmap of all files' median vectors.
    Rows = files, Columns = features.
    """
    plt.figure(figsize=figsize)
    plt.imshow(medians_df.values, aspect="auto", cmap=cmap)
    plt.colorbar(label="Median value")
    plt.xticks(range(len(medians_df.columns)), medians_df.columns, rotation=90)
    plt.yticks(range(len(medians_df)), medians_df.index)
    plt.xlabel("Feature")
    plt.ylabel("File (source_file)")
    plt.title("Heatmap of per-file median vectors")
    plt.tight_layout()
    plt.show()

def filter_outliers_iqr(df, cols, k=1.5):
    """
    Remove rows where any value in `cols` is outside the IQR fence.
    Returns (filtered_df, list_of_removed_files).
    """
    outliers = set()
    for c in cols:
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower = q1 - k * iqr
        upper = q3 + k * iqr
        mask = (df[c] < lower) | (df[c] > upper)
        outliers.update(df.index[mask].tolist())
    filtered_df = df.drop(index=outliers)
    return filtered_df, sorted(outliers)

def filter_outliers_dbscan(df, cols, eps=0.5, min_samples=5):
    """
    Run DBSCAN on the sub-DataFrame df[cols].
    Returns (filtered_df, list_of_noise_files).
    """
    X = df[cols].values
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
    noise_idx = df.index[labels == -1].tolist()
    filtered_df = df.drop(index=noise_idx)
    return filtered_df, sorted(noise_idx)

def create_lagged_dataset(df, y_cols, u_cols, n_lags=5):
    """
    Given a DataFrame for one file, create lagged X and y arrays.

    y_cols = output variables
    u_cols = input variables

    X shape = (num_samples, (len(y_cols) + len(u_cols)) * n_lags)
    y shape = (num_samples, len(y_cols))
    """
    df = df.copy()
    for lag in range(1, n_lags+1):
        for col in y_cols + u_cols:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    df.dropna(inplace=True)

    X_cols = [f"{col}_lag{lag}" for lag in range(1, n_lags+1) for col in y_cols + u_cols]
    X = df[X_cols].values
    y = df[y_cols].values
    return X, y

def compute_stats(df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    """
    Given a DataFrame with a 'source_file' column,
    compute per-file means and medians for all numeric columns.
    Returns two DataFrames (means_df, medians_df) indexed by source_file.
    """
    g = df.groupby('source_file')
    means_df = g.mean()
    medians_df = g.median()
    return means_df, medians_df