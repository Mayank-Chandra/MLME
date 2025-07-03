# clustering.py

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

def prepare_cluster_input(df, use_pca=True, pca_components=2):
    """
    Prepare data for clustering:
      - Standardize numeric cols
      - Optionally reduce dimensions with PCA
    """
    num_cols = df.select_dtypes(include='number').columns
    X = df[num_cols].fillna(0).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if use_pca:
        pca = PCA(n_components=pca_components)
        X_pca = pca.fit_transform(X_scaled)
        return X_pca, scaler, pca
    else:
        return X_scaled, scaler, None

def plot_k_distance_curve(X, k=5):
    """
    Plots the sorted k-distance curve to help choose ε for DBSCAN.
    """
    nbrs = NearestNeighbors(n_neighbors=k)
    nbrs.fit(X)
    distances, _ = nbrs.kneighbors(X)
    k_distances = np.sort(distances[:, -1])

    plt.figure(figsize=(8,3))
    plt.plot(k_distances)
    plt.title(f"{k}-distance plot (choose ε at the knee)")
    plt.xlabel("Points sorted by distance")
    plt.ylabel(f"Distance to {k}th neighbor")
    plt.grid()
    plt.show()

def cluster_dbscan(
    df,
    eps=2.0,
    min_samples=5,
    use_pca=True,
    pca_components=2
):
    """
    Run DBSCAN and return clustering results.
    """
    X, scaler, pca = prepare_cluster_input(df, use_pca=use_pca, pca_components=pca_components)
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X)
    return {
        "labels": labels,
        "cluster_input": X,
        "scaler": scaler,
        "pca": pca
    }

def plot_dbscan_clusters(X, labels):
    """
    Plot clusters in 2D PCA space.
    """
    plt.figure(figsize=(6,4))
    unique_labels = set(labels)
    for label in unique_labels:
        mask = labels == label
        plt.scatter(X[mask,0], X[mask,1], label=f"Cluster {label}", alpha=0.5)
    plt.legend()
    plt.title("DBSCAN clusters")
    plt.show()

def split_data(df, fractions=(0.40, 0.30, 0.15, 0.15), random_state=42):
    """
    Splits df into Train, Val, Cal, Test sets with given fractions.
    """
    assert sum(fractions) == 1.0, "fractions must sum to 1.0"
    df_shuffled = df.sample(frac=1, random_state=random_state)

    n = len(df_shuffled)
    i1 = int(fractions[0] * n)
    i2 = i1 + int(fractions[1] * n)
    i3 = i2 + int(fractions[2] * n)

    train_df = df_shuffled.iloc[:i1].reset_index(drop=True)
    val_df   = df_shuffled.iloc[i1:i2].reset_index(drop=True)
    cal_df   = df_shuffled.iloc[i2:i3].reset_index(drop=True)
    test_df  = df_shuffled.iloc[i3:].reset_index(drop=True)

    return train_df, val_df, cal_df, test_df
