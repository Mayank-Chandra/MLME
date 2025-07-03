# sfc_ml_pipeline.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn

# ----------------------------
# Data Loading
# ----------------------------

def load_all_files(folder_path, sep="\t"):
    data_dict = {}
    for fname in os.listdir(folder_path):
        if fname.endswith(".txt"):
            file_path = os.path.join(folder_path, fname)
            df = pd.read_csv(file_path, sep=sep, engine="python")
            data_dict[fname] = df
    return data_dict

def plot_all_signals(data_dict, n_rows=3, n_cols=2):
    for fname, df in data_dict.items():
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(12, 8))
        axs = axs.flatten()
        for i, col in enumerate(df.columns[:n_rows * n_cols]):
            axs[i].plot(df[col])
            axs[i].set_title(f"{col}")
        plt.suptitle(f"File: {fname}")
        plt.tight_layout()
        plt.show()

# ----------------------------
# Feature Engineering & Clustering
# ----------------------------

def extract_features_per_file(data_dict):
    feature_list = []
    filenames = []
    for fname, df in data_dict.items():
        df_clean = df.dropna()
        features = []
        for col in df_clean.columns:
            features.append(df_clean[col].mean())
            features.append(df_clean[col].std())
            features.append(df_clean[col].iloc[-1])
        feature_list.append(features)
        filenames.append(fname)
    feature_df = pd.DataFrame(feature_list)
    feature_df["filename"] = filenames
    return feature_df

def cluster_features(features, n_clusters=3):
    X = features.drop("filename", axis=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    clusters = kmeans.fit_predict(X_scaled)
    features["cluster"] = clusters
    return features, scaler, kmeans

def plot_clusters(X_scaled, clusters):
    sns.scatterplot(x=X_scaled[:,0], y=X_scaled[:,1], hue=clusters)
    plt.title("Clusters of trajectories")
    plt.show()

# ----------------------------
# Dataset Preparation
# ----------------------------

def create_lagged_dataset(df, y_cols, u_cols, n_lags):
    data = df[y_cols + u_cols].values
    X, y_next = [], []
    for i in range(n_lags, len(data) - 1):
        y_hist = data[i-n_lags:i, :len(y_cols)].flatten()
        u_hist = data[i-n_lags:i, len(y_cols):].flatten()
        X.append(np.concatenate([y_hist, u_hist]))
        y_next.append(data[i+1, :len(y_cols)])
    return np.array(X), np.array(y_next)

# ----------------------------
# PyTorch Models
# ----------------------------

class NARXModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.net(x)

class QuantileLoss(nn.Module):
    def __init__(self, quantile):
        super().__init__()
        self.quantile = quantile

    def forward(self, preds, target):
        error = target - preds
        return torch.mean(
            torch.max(
                self.quantile * error,
                (self.quantile - 1) * error
            )
        )

# ----------------------------
# Training
# ----------------------------

def train_model(X_train, y_train, model, epochs=50, lr=1e-3, batch_size=64, quantile=None):
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    if quantile is None:
        loss_fn = nn.MSELoss()
    else:
        loss_fn = QuantileLoss(quantile)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: loss = {loss.item():.5f}")
    return model

# ----------------------------
# Evaluation
# ----------------------------

def evaluate_model(model, X_test, y_test):
    X_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_tensor = torch.tensor(y_test, dtype=torch.float32)
    pred = model(X_tensor).detach().numpy()
    mse = np.mean((pred - y_test)**2, axis=0)
    mae = np.mean(np.abs(pred - y_test), axis=0)
    return mse, mae, pred

# ----------------------------
# Conformal Prediction
# ----------------------------

def compute_conformity_scores(y_lower, y_upper, y_true):
    scores = np.maximum(y_lower - y_true, y_true - y_upper)
    return scores

def get_conformal_threshold(scores, alpha=0.1):
    threshold = np.quantile(scores, 1 - alpha)
    return threshold

def adjust_intervals(y_lower, y_upper, threshold):
    return y_lower - threshold, y_upper + threshold
