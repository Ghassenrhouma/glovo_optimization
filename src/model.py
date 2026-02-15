"""ML model training and evaluation module."""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier


def train_regression_models(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """Train regression models and return a dict of fitted models."""
    models = {
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "xgboost": XGBRegressor(random_state=42, verbosity=0),
        "lgbm": LGBMRegressor(random_state=42, verbose=-1),
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
    return models


def train_classification_models(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """Train classification models and return a dict of fitted models."""
    models = {
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "xgboost": XGBClassifier(random_state=42, verbosity=0, eval_metric="logloss"),
        "lgbm": LGBMClassifier(random_state=42, verbose=-1),
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
    return models


def evaluate_regression(models: dict, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """Compute RMSE, MAE, R² for each regression model."""
    results = []
    for name, model in models.items():
        preds = model.predict(X_test)
        results.append({
            "Model": name,
            "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
            "MAE": mean_absolute_error(y_test, preds),
            "R2": r2_score(y_test, preds),
        })
    return pd.DataFrame(results).set_index("Model")


def evaluate_classification(models: dict, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """Compute Accuracy, F1, ROC-AUC for each classification model."""
    results = []
    for name, model in models.items():
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else preds
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, preds),
            "F1": f1_score(y_test, preds, average="weighted"),
            "ROC_AUC": roc_auc_score(y_test, proba),
        })
    return pd.DataFrame(results).set_index("Model")


def save_best_model(models: dict, metrics: pd.DataFrame, path: str, metric_col: str = "RMSE") -> str:
    """Save the best model based on the given metric column."""
    if metric_col in ("RMSE", "MAE"):
        best_name = metrics[metric_col].idxmin()
    else:
        best_name = metrics[metric_col].idxmax()

    best_model = models[best_name]
    joblib.dump(best_model, path)
    print(f"Best model: {best_name} (saved to {path})")
    return best_name


def get_feature_importance(model, feature_names: list) -> pd.DataFrame:
    """Return sorted feature importances as a DataFrame."""
    importances = model.feature_importances_
    fi = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    return fi


def predict_delay(model, input_dict: dict) -> float:
    """Predict delivery_time_minutes from a raw feature dictionary."""
    input_df = pd.DataFrame([input_dict])
    prediction = model.predict(input_df)[0]
    return float(prediction)
