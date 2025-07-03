
# modeling.py

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from skorch import NeuralNetRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

def prepare_narx_dataset(df, y_cols, u_cols, n_lags):
    """
    Build X, y for NARX:
    X[k] = [ y_k,…,y_{k-n_lags},  u_k,…,u_{k-n_lags} ],
    y[k] = y_{k+1}
    """
    data = df[y_cols + u_cols].to_numpy()
    N, F = data.shape
    X, y = [], []
    for k in range(n_lags, N - 1):
        past = [data[k - lag] for lag in range(n_lags + 1)]
        X.append(np.hstack(past))
        y.append(data[k + 1, : len(y_cols)])
    return np.vstack(X).astype(np.float32), np.vstack(y).astype(np.float32)

class NARXNet(nn.Module):
    def __init__(self, in_features, hidden_sizes, out_features):
        super().__init__()
        layers = []
        prev = in_features
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, out_features)]
        self.net = nn.Sequential(*layers)

    def forward(self, X):
        return self.net(X)

def grid_search_narx_pytorch(
    train_df,
    cluster_col: str,
    y_cols: list[str],
    u_cols: list[str],
    n_lags: int,
    param_grid: dict,
    cv_splits: int = 3,
    device: str = 'cuda'
) -> dict[int, dict]:
    """
    Run TimeSeriesSplit + GridSearchCV on each cluster's train_df subset.
    Returns best_params per cluster.
    """
    cluster_params = {}
    tscv = TimeSeriesSplit(n_splits=cv_splits)

    for cid in sorted(train_df[cluster_col].unique()):
        df_c = train_df[train_df[cluster_col] == cid].sort_index().reset_index(drop=True)
        X, y = prepare_narx_dataset(df_c, y_cols, u_cols, n_lags)

        net = NeuralNetRegressor(
            module=NARXNet,
            module__in_features=X.shape[1],
            module__hidden_sizes=(50, 50),  # dummy default, overwritten by grid
            module__out_features=y.shape[1],
            max_epochs=500,                 # dummy default, overwritten by grid
            optimizer=torch.optim.Adam,
            criterion=nn.MSELoss,
            device=device,
            train_split=None,
            verbose=0
        )

        gs = GridSearchCV(
            estimator=net,
            param_grid=param_grid,
            cv=tscv,
            scoring='r2',
            n_jobs=1,
            refit=False
        )
        gs.fit(X, y)
        cluster_params[cid] = gs.best_params_

    return cluster_params

def train_narx_pytorch(
    train_df,
    val_df,
    test_df,
    cluster_col: str,
    y_cols: list[str],
    u_cols: list[str],
    n_lags: int,
    cluster_params: dict[int, dict],
    device: str = 'cuda'
) -> dict[int, dict]:
    """
    For each cluster:
      • fit on train+val with best_params
      • evaluate on test
    Returns dict[cid] = {
      'model', 'train_score', 'test_score', 'X_test', 'y_test'
    }
    """
    results = {}

    for cid, best in cluster_params.items():
        df_tv = pd.concat([
            train_df[train_df[cluster_col] == cid],
            val_df[val_df[cluster_col] == cid]
        ]).sort_index().reset_index(drop=True)
        df_te = test_df[test_df[cluster_col] == cid].sort_index().reset_index(drop=True)

        X_tv, y_tv = prepare_narx_dataset(df_tv, y_cols, u_cols, n_lags)
        X_te, y_te = prepare_narx_dataset(df_te, y_cols, u_cols, n_lags)

        net = NeuralNetRegressor(
            module=NARXNet,
            module__in_features=X_tv.shape[1],
            module__hidden_sizes=best['module__hidden_sizes'],
            module__out_features=y_tv.shape[1],
            max_epochs=best['max_epochs'],
            optimizer=torch.optim.Adam,
            optimizer__lr=best.get('optimizer__lr', 1e-3),
            criterion=nn.MSELoss,
            device=device,
            train_split=None,
            verbose=0
        )

        net.fit(X_tv, y_tv)

        results[cid] = {
            'model': net,
            'train_score': net.score(X_tv, y_tv),
            'test_score': net.score(X_te, y_te),
            'X_test': X_te,
            'y_test': y_te
        }

    return results
