from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def evaluate_model(model, X_test, y_test):
    """Evaluate a model on test data and return classification metrics."""
    y_pred = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }
