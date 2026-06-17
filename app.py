import logging
import os
from collections import deque
from datetime import datetime
from typing import Literal

import mlflow
import mlflow.sklearn
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator


MODEL_URI = "models:/churn-classifier@latest-cindy"

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("logs/api.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("churn_api")

metrics_state = {
    "n_requests": 0,
    "n_errors": 0,
    "n_invalid_inputs": 0,
    "drift_alerts": 0,
    "predictions_distribution": {"churn": 0, "no_churn": 0},
}

TRAIN_RANGES = {
    "age": (22, 61),
    "tenure_months": (1, 72),
    "monthly_charges": (39.5, 95.0),
    "support_calls": (0, 7),
}

TRAIN_STATS = {
    "tenure_months": {"mean": 23.037, "std": 20.1852},
    "monthly_charges": {"mean": 68.0148, "std": 16.0022},
}
WINDOW_SIZE = 50
recent_inputs = {var: deque(maxlen=WINDOW_SIZE) for var in TRAIN_STATS}
REFERENCE_SAMPLE_PATH = "artifacts/reference_sample.csv"
DRIFT_REPORT_BUFFER_SIZE = 100
drift_buffer = deque(maxlen=DRIFT_REPORT_BUFFER_SIZE)

try:
    reference_sample = pd.read_csv(REFERENCE_SAMPLE_PATH)
    logger.info("Echantillon de reference charge depuis %s", REFERENCE_SAMPLE_PATH)
except FileNotFoundError:
    reference_sample = None
    logger.warning(
        "Echantillon de reference introuvable: %s. Lance python main.py.",
        REFERENCE_SAMPLE_PATH,
    )

mlflow.set_tracking_uri("sqlite:///mlflow.db")

try:
    model = mlflow.sklearn.load_model(MODEL_URI)
    model_load_error = None
    logger.info("Modele charge depuis %s", MODEL_URI)
except Exception as exc:
    model = None
    model_load_error = str(exc)
    logger.critical(
        "Service degrade: impossible de charger le modele depuis %s",
        MODEL_URI,
        exc_info=True,
    )


class CustomerInput(BaseModel):
    age: int = Field(..., ge=0, le=120, description="Age du client")
    tenure_months: int = Field(..., ge=0, le=120, description="Anciennete en mois")
    monthly_charges: float = Field(..., ge=0, description="Montant mensuel facture")
    support_calls: int = Field(..., ge=0, le=100, description="Nombre d'appels support")
    contract_type: Literal["Monthly", "One year", "Two year"] = Field(
        ..., description="Type de contrat"
    )
    payment_method: Literal[
        "Bank transfer",
        "Credit card",
        "Electronic check",
        "Mailed check",
    ] = Field(..., description="Methode de paiement")
    internet_service: Literal["DSL", "Fiber"] = Field(
        ..., description="Type de service internet"
    )

    @model_validator(mode="after")
    def check_charges_coherence(self):
        if self.tenure_months > 0 and self.monthly_charges == 0:
            raise ValueError(
                "monthly_charges = 0 incoherent avec tenure_months > 0"
            )
        return self


app = FastAPI(title="Churn Prediction API", version="1.0")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    if request.url.path == "/predict":
        metrics_state["n_requests"] += 1
        metrics_state["n_invalid_inputs"] += 1
        logger.warning("Input invalide sur /predict: %s", exc.errors())

    return JSONResponse(
        status_code=422,
        content=jsonable_encoder({"detail": exc.errors()}),
    )


def _customer_to_dataframe(customer: CustomerInput) -> pd.DataFrame:
    if not hasattr(model, "feature_names_in_"):
        raise HTTPException(
            status_code=500,
            detail="Le modele charge ne contient pas feature_names_in_.",
        )

    payload = (
        customer.model_dump()
        if hasattr(customer, "model_dump")
        else customer.dict()
    )
    df = pd.DataFrame([payload])
    df_encoded = pd.get_dummies(df, drop_first=True)
    return df_encoded.reindex(columns=model.feature_names_in_, fill_value=0)


def detect_anomaly(payload: CustomerInput) -> list[str]:
    anomalies = []
    for var, (low, high) in TRAIN_RANGES.items():
        value = getattr(payload, var)
        if value < low or value > high:
            anomalies.append(f"{var}={value} hors plage [{low}, {high}]")
    return anomalies


def check_drift(payload: CustomerInput) -> dict:
    drift_report = {}
    for var, stats in TRAIN_STATS.items():
        recent_inputs[var].append(getattr(payload, var))
        if len(recent_inputs[var]) >= WINDOW_SIZE:
            recent_mean = sum(recent_inputs[var]) / len(recent_inputs[var])
            gap = abs(recent_mean - stats["mean"]) / stats["std"]
            if gap > 0.5:
                drift_report[var] = round(gap, 2)
    return drift_report


@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/health")
def health():
    logger.info("Health check: model_loaded=%s", model is not None)
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "model_uri": MODEL_URI,
    }


@app.get("/metrics")
def metrics():
    return metrics_state


@app.post("/drift_report")
def drift_report():
    if reference_sample is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"{REFERENCE_SAMPLE_PATH} introuvable. "
                "Relance python main.py pour generer l'echantillon de reference."
            ),
        )

    if len(drift_buffer) < DRIFT_REPORT_BUFFER_SIZE:
        raise HTTPException(
            status_code=425,
            detail=(
                f"Buffer : {len(drift_buffer)}/{DRIFT_REPORT_BUFFER_SIZE}. "
                "Envoie plus de requetes /predict."
            ),
        )

    current = pd.DataFrame(list(drift_buffer))
    result = Report(metrics=[DataDriftPreset()]).run(
        reference_data=reference_sample,
        current_data=current,
    )

    os.makedirs("logs/drift_reports", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"logs/drift_reports/drift_{timestamp}.html"
    result.save_html(path)

    metrics_state.setdefault("evidently_reports_generated", 0)
    metrics_state["evidently_reports_generated"] += 1
    logger.info("Rapport Evidently genere: %s", path)
    return {"status": "report_generated", "report_path": path}


@app.post("/predict")
def predict(payload: CustomerInput):
    metrics_state["n_requests"] += 1
    logger.info("Requete recue : tenure_months=%s", payload.tenure_months)

    anomalies = detect_anomaly(payload)
    if anomalies:
        metrics_state["n_invalid_inputs"] += 1
        metrics_state["drift_alerts"] += 1
        logger.warning("Entree anormale : %s", anomalies)
        return {
            "prediction": None,
            "warning": "anomalie detectee",
            "details": anomalies,
        }

    drift = check_drift(payload)
    if drift:
        logger.warning("Derive detectee sur %s requetes : %s", WINDOW_SIZE, drift)
        metrics_state["drift_alerts"] += 1

    if model is None:
        metrics_state["n_errors"] += 1
        logger.warning("Prediction refusee: modele indisponible: %s", model_load_error)
        raise HTTPException(
            status_code=503,
            detail=f"Modele indisponible: {model_load_error}",
        )

    try:
        df_aligned = _customer_to_dataframe(payload)
        drift_buffer.append(df_aligned.iloc[0].to_dict())
        prediction = int(model.predict(df_aligned)[0])
        proba = float(model.predict_proba(df_aligned)[0].max())
        label = "churn" if prediction == 1 else "no_churn"
        metrics_state["predictions_distribution"][label] += 1
        logger.info("Prediction : %s (confiance=%.2f)", prediction, proba)
        return {
            "prediction": prediction,
            "label": label,
            "confidence": proba,
        }
    except Exception as e:
        metrics_state["n_errors"] += 1
        logger.error("Echec de prediction : %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
