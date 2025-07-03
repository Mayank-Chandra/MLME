# data_utils.py

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def load_txt_data(folder_path: str, sep="\t") -> pd.DataFrame:
    """
    Loads all .txt files from the given folder_path
    into a single concatenated DataFrame.
    """
    base_path = Path(folder_path)
    txt_files = sorted(base_path.rglob("*.txt"))

    dfs = []
    for file in txt_files:
        df = pd.read_csv(file, sep=sep)
        df["source_file"] = file.name
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)

def plot_feature_distributions(df: pd.DataFrame, bins: int = 50) -> None:
    """
    Plots histograms for all numeric columns.
    """
    num_cols = df.select_dtypes(include='number').columns
    for col in num_cols:
        plt.figure(figsize=(6,3))
        plt.hist(df[col].dropna(), bins=bins, edgecolor='black')
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.show()
