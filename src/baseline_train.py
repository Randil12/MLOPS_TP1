import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from helper import load_dataset, split_features_target, compute_metrics, save_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "churn.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
METRICS_PATH = ARTIFACTS_DIR / "metrics.txt"

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

df = load_dataset(DATA_PATH)

X, y = split_features_target(df, target_column="churn")

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

metrics = compute_metrics(y_test, y_pred)

print("Evaluation metrics:")
for key, value in metrics.items():
    print(f"{key}: {value:.4f}")

save_metrics(metrics, METRICS_PATH)

with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

print(f"Model saved to {MODEL_PATH}")
print(f"Metrics saved to {METRICS_PATH}")
