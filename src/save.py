import pickle
from pathlib import Path


def _set_registered_model_alias(model_name, alias, model_info):
    """Point a Registry alias to the version created by the current MLflow run."""
    import mlflow
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    version = getattr(model_info, "registered_model_version", None)

    if version is None:
        run_id = mlflow.active_run().info.run_id
        versions = client.search_model_versions(
            f"name = '{model_name}' and run_id = '{run_id}'"
        )
        if not versions:
            raise RuntimeError(
                f"Aucune version MLflow trouvee pour le modele '{model_name}'."
            )
        version = max(int(model_version.version) for model_version in versions)

    client.set_registered_model_alias(model_name, alias, str(version))
    return str(version)


def save_artifacts(
    model,
    metrics,
    model_path,
    metrics_path,
    X_train=None,
    registered_model_name=None,
    model_alias=None,
):
    """Save the trained model and metrics to the provided output paths."""
    model_path = Path(model_path)
    metrics_path = Path(metrics_path)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    if X_train is not None:
        sample_size = min(1000, len(X_train))
        X_train.sample(n=sample_size, random_state=42).to_csv(
            model_path.parent / "reference_sample.csv",
            index=False,
        )

    with open(metrics_path, "w", encoding="utf-8") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value:.4f}\n")

    if not model_path.exists():
        raise RuntimeError(f"Le modele n'a pas ete sauvegarde : {model_path}")

    if not metrics_path.exists():
        raise RuntimeError(f"Les metriques n'ont pas ete sauvegardees : {metrics_path}")

    registered_model_version = None

    if registered_model_name is not None:
        import mlflow.sklearn

        model_info = mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=registered_model_name,
        )
        if model_alias is not None:
            registered_model_version = _set_registered_model_alias(
                registered_model_name,
                model_alias,
                model_info,
            )

    return model_path, metrics_path, registered_model_version
