"""Preprocessing module for Glovo delivery optimization."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Constants
DROP_COLS = ["Unnamed: 0", "delivery_status_y", "promised_time", "actual_time"]

RENAME_COLS = {
    "delivery_status_x": "delivery_status",
    "total_Quantity": "total_quantity",
}

STATUS_MAP = {
    "On Time": 0,
    "Slightly Delayed": 1,
    "Significantly Delayed": 2,
}

FEATURE_COLS = [
    "order_total",
    "number_product",
    "total_quantity",
    "distance_km",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "is_peak_hour",
    "month",
    "payment_method_encoded",
    "customer_segment_encoded",
    "total_orders",
    "avg_order_value",
    "profit_margin",
]

TARGET_REGRESSION = "delivery_time_minutes"
TARGET_CLASSIFICATION = "is_delayed"

DATETIME_COLS = [
    "order_date",
    "promised_delivery_time",
    "actual_delivery_time",
    "feedback_date",
    "registration_date",
]


def load_data(filepath: str) -> pd.DataFrame:
    """Load and clean the raw orders CSV file."""
    df = pd.read_csv(filepath)

    # Drop unnecessary columns (only if they exist)
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    # Rename columns
    rename = {k: v for k, v in RENAME_COLS.items() if k in df.columns}
    df = df.rename(columns=rename)

    # Parse datetime columns
    for col in DATETIME_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer new features from the cleaned DataFrame."""
    df = df.copy()

    # Time-based features from order_date
    df["hour_of_day"] = df["order_date"].dt.hour
    df["day_of_week"] = df["order_date"].dt.dayofweek  # 0=Monday
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_peak_hour"] = (
        ((df["hour_of_day"] >= 11) & (df["hour_of_day"] <= 14))
        | ((df["hour_of_day"] >= 18) & (df["hour_of_day"] <= 21))
    ).astype(int)
    df["month"] = df["order_date"].dt.month

    # Delay flags
    df["is_delayed"] = (df["delivery_status"] != "On Time").astype(int)
    df["is_significantly_delayed"] = (
        df["delivery_status"] == "Significantly Delayed"
    ).astype(int)

    # Customer lifetime
    df["customer_lifetime_days"] = (
        df["order_date"] - df["registration_date"]
    ).dt.days

    # Profit margin (handle divide-by-zero)
    df["profit_margin"] = np.where(
        df["order_total"] != 0,
        df["total_profit"] / df["order_total"],
        0.0,
    )

    # Order size category
    df["order_size_category"] = pd.cut(
        df["order_total"],
        bins=[-np.inf, 500, 2000, np.inf],
        labels=["Small", "Medium", "Large"],
    )

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode categorical columns with _encoded suffix."""
    df = df.copy()

    # Label encode these columns
    label_cols = ["payment_method", "customer_segment", "feedback_category", "sentiment"]
    for col in label_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[f"{col}_encoded"] = le.fit_transform(df[col].astype(str))

    # Map delivery_status explicitly
    if "delivery_status" in df.columns:
        df["delivery_status_encoded"] = df["delivery_status"].map(STATUS_MAP)

    return df


def get_feature_matrix(df: pd.DataFrame) -> tuple:
    """Return (X, y) for regression on delivery_time_minutes."""
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].copy()
    y = df[TARGET_REGRESSION].copy()
    return X, y


def get_classification_matrix(df: pd.DataFrame) -> tuple:
    """Return (X, y) for binary classification on is_delayed."""
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].copy()
    y = df[TARGET_CLASSIFICATION].copy()
    return X, y
