from pathlib import Path

import mlflow
import mlflow.sklearn

from src.evaluate import evaluate_model
from src.prepare import prepare_data
from src.save import save_artifacts
from src.train import train_model


def main():
    """Run the explicit ML pipeline from preparation to artifact logging."""
    data_path = Path("data/raw/churn.csv")
    artifacts_dir = Path("artifacts")
    processed_dir = Path("data/processed")
    model_path = artifacts_dir / "model.pkl"
    metrics_path = artifacts_dir / "metrics.txt"
    validation_path = processed_dir / "validation.csv"
    registered_model_name = "churn-classifier"
    model_alias = "latest-cindy"

    n_estimators = 200
    max_depth = 5
    test_size = 0.2
    random_state = 42
    target_column = "churn"

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("churn-baseline")

    with mlflow.start_run():
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)

        # Etape 1 : preparation
        X_train, X_test, y_train, y_test = prepare_data(
            data_path=data_path,
            target_column=target_column,
            test_size=test_size,
            random_state=random_state,
        )
        processed_dir.mkdir(parents=True, exist_ok=True)
        X_test.assign(churn=y_test.values).to_csv(validation_path, index=False)

        # Etape 2 : entrainement
        model = train_model(
            X_train=X_train,
            y_train=y_train,
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
        )

        # Etape 3 : evaluation
        metrics = evaluate_model(
            model=model,
            X_test=X_test,
            y_test=y_test,
        )

        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")
            mlflow.log_metric(key, value)

        # Etape 4 : sauvegarde
        saved_model_path, saved_metrics_path, registered_model_version = save_artifacts(
            model=model,
            metrics=metrics,
            model_path=model_path,
            metrics_path=metrics_path,
            X_train=X_train,
            registered_model_name=registered_model_name,
            model_alias=model_alias,
        )

        mlflow.log_artifact(str(saved_metrics_path))
        mlflow.log_artifact(str(saved_model_path))

        model_uri = f"models:/{registered_model_name}@{model_alias}"
        reloaded_model = mlflow.sklearn.load_model(model_uri)
        print(f"Modele recharge depuis {model_uri}: {type(reloaded_model)}")
        if registered_model_version is not None:
            print(
                f"Alias {model_alias} -> {registered_model_name} "
                f"version {registered_model_version}"
            )


if __name__ == "__main__":
    main()
