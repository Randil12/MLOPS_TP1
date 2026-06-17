from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def prepare_data(
    data_path,
    target_column,
    test_size=0.2,
    random_state=42,
):
    """Load data, encode features, and return train/test splits."""
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset introuvable : {data_path}")

    df = pd.read_csv(data_path, sep=",")

    if df.empty:
        raise ValueError("Le dataset est vide.")

    if target_column not in df.columns:
        raise KeyError(f"Colonne cible '{target_column}' absente.")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X = pd.get_dummies(X, drop_first=True)

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
