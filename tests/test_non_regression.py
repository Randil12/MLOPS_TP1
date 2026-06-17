import pickle

import pandas as pd
from sklearn.metrics import accuracy_score

import app as api


ACCURACY_THRESHOLD = 0.75


def test_model_accuracy_above_threshold():
    with open("artifacts/model.pkl", "rb") as f:
        model = pickle.load(f)

    df_val = pd.read_csv("data/processed/validation.csv")
    X_val = df_val.drop(columns=["churn"])
    y_val = df_val["churn"]

    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)

    assert accuracy >= ACCURACY_THRESHOLD, (
        f"Regression detectee : accuracy={accuracy:.3f} < "
        f"seuil={ACCURACY_THRESHOLD}"
    )


def test_customer_dataframe_keeps_training_feature_order(monkeypatch):
    class ModelWithFeatures:
        feature_names_in_ = [
            "age",
            "tenure_months",
            "monthly_charges",
            "support_calls",
            "contract_type_One year",
            "contract_type_Two year",
            "payment_method_Credit card",
            "payment_method_Electronic check",
            "payment_method_Mailed check",
            "internet_service_Fiber",
        ]

    monkeypatch.setattr(api, "model", ModelWithFeatures())

    payload = api.CustomerInput(
        age=35,
        tenure_months=12,
        monthly_charges=75.5,
        support_calls=2,
        contract_type="Monthly",
        payment_method="Electronic check",
        internet_service="Fiber",
    )

    df = api._customer_to_dataframe(payload)

    assert list(df.columns) == ModelWithFeatures.feature_names_in_
    assert df.loc[0, "age"] == 35
    assert df.loc[0, "tenure_months"] == 12
    assert df.loc[0, "monthly_charges"] == 75.5
    assert not df.loc[0, "contract_type_One year"]
