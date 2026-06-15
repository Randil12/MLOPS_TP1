from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def load_dataset(path):
    df = pd.read_csv(path, sep=",")
    return df


def split_features_target(df, target_column="churn"):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


def compute_metrics(y_true, y_pred):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }
    return metrics


def save_metrics(metrics, output_path="metrics.txt"):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value:.4f}\n")
