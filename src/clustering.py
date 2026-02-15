"""Area-based clustering and performance analysis module."""

import pandas as pd
import numpy as np


def compute_area_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute aggregated statistics per delivery area."""
    stats = df.groupby("area").agg(
        order_count=("order_id", "count"),
        avg_delivery_delay=("delivery_time_minutes", "mean"),
        delay_rate_pct=("is_delayed", lambda x: round(x.mean() * 100, 2)),
        significant_delay_rate_pct=("is_significantly_delayed", lambda x: round(x.mean() * 100, 2)),
        avg_distance_km=("distance_km", "mean"),
        avg_order_total=("order_total", "mean"),
        avg_rating=("rating", "mean"),
        positive_sentiment_pct=("sentiment", lambda x: round((x == "Positive").mean() * 100, 2)),
    ).reset_index()

    stats = stats.sort_values("delay_rate_pct", ascending=False).reset_index(drop=True)
    return stats


def classify_area_performance(area_stats: pd.DataFrame) -> pd.DataFrame:
    """Classify areas into High/Medium/Low Risk tiers based on delay rate."""
    df = area_stats.copy()
    conditions = [
        df["delay_rate_pct"] > 40,
        df["delay_rate_pct"] >= 20,
    ]
    choices = ["High Risk", "Medium Risk"]
    df["performance_tier"] = np.select(conditions, choices, default="Low Risk")
    return df


def get_top_bottom_areas(area_stats: pd.DataFrame, n: int = 10) -> tuple:
    """Return (best_n, worst_n) areas by delay rate."""
    sorted_df = area_stats.sort_values("delay_rate_pct")
    best = sorted_df.head(n).reset_index(drop=True)
    worst = sorted_df.tail(n).iloc[::-1].reset_index(drop=True)
    return best, worst


def compute_hourly_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute average delay and delay rate per hour of day."""
    hourly = df.groupby("hour_of_day").agg(
        avg_delivery_delay=("delivery_time_minutes", "mean"),
        delay_rate_pct=("is_delayed", lambda x: round(x.mean() * 100, 2)),
        order_count=("order_id", "count"),
    ).reset_index()
    return hourly


def compute_segment_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Compute summary statistics per customer segment."""
    seg = df.groupby("customer_segment").agg(
        avg_order_value=("order_total", "mean"),
        avg_rating=("rating", "mean"),
        delay_rate_pct=("is_delayed", lambda x: round(x.mean() * 100, 2)),
        avg_total_orders=("total_orders", "mean"),
        order_count=("order_id", "count"),
    ).reset_index()
    return seg
