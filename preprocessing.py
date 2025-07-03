# preprocessing.py

import pandas as pd
import numpy as np

def preprocess_features(
    df: pd.DataFrame,
    corr_threshold: float = 0.95,
    consider_only_numeric: bool = True
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """
    Drops:
      - constant columns
      - highly correlated columns (above threshold)
    """
    removed = {
        "constant": [],
        "highly_correlated": []
    }

    if consider_only_numeric:
        cols = df.select_dtypes(include='number').columns
    else:
        cols = df.columns

    # Drop constant cols
    for col in cols:
        if df[col].nunique() <= 1:
            removed["constant"].append(col)

    df = df.drop(columns=removed["constant"], errors='ignore')

    # Drop highly correlated cols
    corr_matrix = df[cols].corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    to_drop = [
        column
        for column in upper.columns
        if any(upper[column] > corr_threshold)
    ]

    removed["highly_correlated"] = to_drop
    df = df.drop(columns=to_drop, errors='ignore')

    return df, removed
