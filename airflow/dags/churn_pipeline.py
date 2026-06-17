import os
import pickle
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator

PROJECT_ROOT = Path("/opt/airflow/project")
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate import evaluate_model  # noqa: E402
from src.prepare import prepare_data  # noqa: E402
from src.save import save_artifacts  # noqa: E402
from src.train import train_model  # noqa: E402


DATA_PATH = PROJECT_ROOT / "data" / "raw" / "churn.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

X_TRAIN_PATH = PROCESSED_DIR / "X_train.csv"
X_TEST_PATH = PROCESSED_DIR / "X_test.csv"
Y_TRAIN_PATH = PROCESSED_DIR / "y_train.csv"
Y_TEST_PATH = PROCESSED_DIR / "y_test.csv"
VALIDATION_PATH = PROCESSED_DIR / "validation.csv"
MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
METRICS_PATH = ARTIFACTS_DIR / "metrics.txt"


def task_check_data(**context):
    csv_path = str(DATA_PATH)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset introuvable : {csv_path}")
    return csv_path


def task_prepare_data(**context):
    csv_path = context["ti"].xcom_pull(task_ids="check_data")
    X_train, X_test, y_train, y_test = prepare_data(
        data_path=csv_path,
        target_column="churn",
        test_size=0.2,
        random_state=42,
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(X_TRAIN_PATH, index=False)
    X_test.to_csv(X_TEST_PATH, index=False)
    y_train.to_frame("churn").to_csv(Y_TRAIN_PATH, index=False)
    y_test.to_frame("churn").to_csv(Y_TEST_PATH, index=False)
    X_test.assign(churn=y_test.values).to_csv(VALIDATION_PATH, index=False)

    return {
        "X_train_path": str(X_TRAIN_PATH),
        "X_test_path": str(X_TEST_PATH),
        "y_train_path": str(Y_TRAIN_PATH),
        "y_test_path": str(Y_TEST_PATH),
        "validation_path": str(VALIDATION_PATH),
    }


def task_train_model(**context):
    paths = context["ti"].xcom_pull(task_ids="prepare_data")
    X_train = pd.read_csv(paths["X_train_path"])
    y_train = pd.read_csv(paths["y_train_path"])["churn"]

    model = train_model(
        X_train=X_train,
        y_train=y_train,
        n_estimators=200,
        max_depth=5,
        random_state=42,
    )

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    return str(MODEL_PATH)


def task_evaluate_model(**context):
    paths = context["ti"].xcom_pull(task_ids="prepare_data")
    model_path = context["ti"].xcom_pull(task_ids="train_model")

    X_test = pd.read_csv(paths["X_test_path"])
    y_test = pd.read_csv(paths["y_test_path"])["churn"]

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    metrics = evaluate_model(model, X_test, y_test)
    context["ti"].xcom_push(key="metrics", value=metrics)
    return metrics


def task_save_artifacts(**context):
    paths = context["ti"].xcom_pull(task_ids="prepare_data")
    model_path = context["ti"].xcom_pull(task_ids="train_model")
    metrics = context["ti"].xcom_pull(task_ids="evaluate_model", key="metrics")

    X_train = pd.read_csv(paths["X_train_path"])

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    saved_model_path, saved_metrics_path, _ = save_artifacts(
        model=model,
        metrics=metrics,
        model_path=MODEL_PATH,
        metrics_path=METRICS_PATH,
        X_train=X_train,
    )

    return {
        "model_path": str(saved_model_path),
        "metrics_path": str(saved_metrics_path),
    }


default_args = {
    "owner": "etudiant",
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    dag_id="churn_pipeline",
    default_args=default_args,
    schedule="*/5 * * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["mlops", "tp-jour2"],
) as dag:
    check = PythonOperator(
        task_id="check_data",
        python_callable=task_check_data,
        retries=3,
        retry_delay=timedelta(minutes=1),
    )
    prepare = PythonOperator(
        task_id="prepare_data",
        python_callable=task_prepare_data,
    )
    train = PythonOperator(
        task_id="train_model",
        python_callable=task_train_model,
    )
    evaluate = PythonOperator(
        task_id="evaluate_model",
        python_callable=task_evaluate_model,
    )
    save = PythonOperator(
        task_id="save_artifacts",
        python_callable=task_save_artifacts,
    )

    check >> prepare >> train >> evaluate >> save
