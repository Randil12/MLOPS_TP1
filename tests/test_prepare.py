import pandas as pd
import pytest

from src.prepare import prepare_data


def test_prepare_data_returns_four_objects(tmp_path):
    df = pd.DataFrame(
        {
            "tenure": [1, 12, 24, 36, 48, 60],
            "monthly_charges": [20.0, 50.0, 70.0, 80.0, 90.0, 100.0],
            "churn": [0, 1, 0, 1, 0, 1],
        }
    )
    csv_path = tmp_path / "mini.csv"
    df.to_csv(csv_path, index=False)

    result = prepare_data(str(csv_path), target_column="churn")

    assert len(result) == 4


def test_prepare_data_raises_on_missing_file():
    with pytest.raises(FileNotFoundError):
        prepare_data("inexistant.csv", target_column="churn")


def test_prepare_data_raises_on_missing_target(tmp_path):
    df = pd.DataFrame({"tenure": [1, 2], "monthly_charges": [20.0, 30.0]})
    csv_path = tmp_path / "no_target.csv"
    df.to_csv(csv_path, index=False)

    with pytest.raises(KeyError):
        prepare_data(str(csv_path), target_column="churn")


def test_prepare_data_encodes_categorical_columns(tmp_path):
    data_path = tmp_path / "churn.csv"
    pd.DataFrame(
        [
            {
                "age": 24,
                "tenure_months": 2,
                "monthly_charges": 79.9,
                "contract_type": "Monthly",
                "payment_method": "Electronic check",
                "internet_service": "Fiber",
                "support_calls": 4,
                "churn": 1,
            },
            {
                "age": 45,
                "tenure_months": 36,
                "monthly_charges": 54.2,
                "contract_type": "One year",
                "payment_method": "Bank transfer",
                "internet_service": "DSL",
                "support_calls": 1,
                "churn": 0,
            },
            {
                "age": 31,
                "tenure_months": 6,
                "monthly_charges": 88.5,
                "contract_type": "Monthly",
                "payment_method": "Electronic check",
                "internet_service": "Fiber",
                "support_calls": 5,
                "churn": 1,
            },
            {
                "age": 52,
                "tenure_months": 48,
                "monthly_charges": 49.8,
                "contract_type": "Two year",
                "payment_method": "Credit card",
                "internet_service": "DSL",
                "support_calls": 0,
                "churn": 0,
            },
        ]
    ).to_csv(data_path, index=False)

    X_train, X_test, y_train, y_test = prepare_data(
        data_path=data_path,
        target_column="churn",
        test_size=0.5,
        random_state=42,
    )

    assert "churn" not in X_train.columns
    assert "contract_type_One year" in X_train.columns
    assert len(X_train) == 2
    assert len(X_test) == 2
    assert set(y_train.unique()) == {0, 1}
    assert set(y_test.unique()) == {0, 1}


def test_prepare_data_raises_when_target_missing(tmp_path):
    data_path = tmp_path / "churn.csv"
    pd.DataFrame({"age": [24], "churn_missing": [1]}).to_csv(data_path, index=False)

    with pytest.raises(KeyError):
        prepare_data(
            data_path=data_path,
            target_column="churn",
            test_size=0.2,
            random_state=42,
        )


def test_echec_volontaire_ci():
    assert False
