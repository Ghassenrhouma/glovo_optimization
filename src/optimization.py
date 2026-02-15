"""Courier zone load balancing and staffing optimization module."""

import pandas as pd
import numpy as np
from math import ceil


def compute_zone_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Compute courier zone (area-based) performance stats.

    Since each delivery_partner_id appears only once in the dataset,
    we group by area to create meaningful 'courier zones' that reflect
    how couriers operating in each zone perform collectively.
    """
    stats = df.groupby("area").agg(
        total_deliveries=("order_id", "count"),
        avg_delay_minutes=("delivery_time_minutes", "mean"),
        on_time_rate_pct=("is_delayed", lambda x: round((1 - x.mean()) * 100, 2)),
        avg_distance_km=("distance_km", "mean"),
        avg_order_total=("order_total", "mean"),
        avg_rating=("rating", "mean"),
    ).reset_index()

    stats = stats.sort_values("on_time_rate_pct", ascending=True).reset_index(drop=True)
    return stats


def identify_overloaded_zones(zone_stats: pd.DataFrame, on_time_threshold: float = 50.0) -> pd.DataFrame:
    """Flag zones with on-time rate below the given threshold."""
    return zone_stats[zone_stats["on_time_rate_pct"] < on_time_threshold].reset_index(drop=True)


def simulate_load_redistribution(df: pd.DataFrame) -> dict:
    """Simulate redistributing orders from high-delay zones to lower-delay ones.

    Logic: identify zones with above-average delay, assume 20 % of their orders
    can be shifted to nearby lower-delay zones, and estimate the fleet-wide
    improvement.
    """
    zone_stats = compute_zone_performance(df)
    avg_delay = df["delivery_time_minutes"].mean()

    high_delay_zones = zone_stats[zone_stats["avg_delay_minutes"] > avg_delay]
    low_delay_zones = zone_stats[zone_stats["avg_delay_minutes"] <= avg_delay]

    before_avg_delay = avg_delay

    # Orders to redistribute: 20 % of high-delay zone volume
    orders_to_redistribute = int(high_delay_zones["total_deliveries"].sum() * 0.20)

    # Assume redistributed orders achieve the average delay of low-delay zones
    low_delay_avg = low_delay_zones["avg_delay_minutes"].mean() if len(low_delay_zones) > 0 else avg_delay
    remaining_orders = len(df) - orders_to_redistribute
    after_avg_delay = (
        (remaining_orders * avg_delay + orders_to_redistribute * low_delay_avg)
        / len(df)
    )

    improvement_pct = ((before_avg_delay - after_avg_delay) / abs(before_avg_delay) * 100
                       if before_avg_delay != 0 else 0.0)

    return {
        "before_avg_delay": round(before_avg_delay, 2),
        "after_avg_delay": round(after_avg_delay, 2),
        "improvement_pct": round(abs(improvement_pct), 2),
        "orders_redistributed": orders_to_redistribute,
        "high_delay_zones": len(high_delay_zones),
        "low_delay_zones": len(low_delay_zones),
    }


def compute_peak_hour_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Compute order volume, delay, and courier utilization per hour."""
    hourly = df.groupby("hour_of_day").agg(
        order_count=("order_id", "count"),
        avg_delay_minutes=("delivery_time_minutes", "mean"),
        unique_couriers=("delivery_partner_id", "nunique"),
    ).reset_index()

    hourly["utilization"] = (hourly["order_count"] / hourly["unique_couriers"]).round(2)
    return hourly


def recommend_staffing(peak_df: pd.DataFrame) -> pd.DataFrame:
    """Recommend extra couriers for the most understaffed hours."""
    df = peak_df.copy()

    # Score by combination of utilization and delay
    df["stress_score"] = df["utilization"] * df["avg_delay_minutes"]
    top = df.nlargest(5, "stress_score").copy()

    top["recommended_extra"] = top.apply(
        lambda row: max(0, ceil(row["order_count"] / 10) - int(row["unique_couriers"])),
        axis=1,
    )

    # Estimate delay reduction: proportional to extra capacity added
    top["expected_delay_reduction_min"] = (
        top["recommended_extra"] / (top["unique_couriers"] + top["recommended_extra"])
        * top["avg_delay_minutes"]
    ).round(2)

    result = top[
        ["hour_of_day", "unique_couriers", "recommended_extra", "expected_delay_reduction_min"]
    ].rename(columns={"unique_couriers": "current_couriers"}).reset_index(drop=True)

    return result
