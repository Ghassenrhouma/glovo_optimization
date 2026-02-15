"""CO2 emission estimation module based on delivery distance."""

import pandas as pd
import numpy as np

# Emission factors in gCO2 per km
EMISSION_FACTORS = {
    "motorcycle": 103,
    "bicycle": 0,
    "electric_scooter": 22,
    "car": 171,
}


def estimate_co2_per_order(distance_km: float, mode: str = "motorcycle") -> float:
    """Estimate CO2 emissions in grams for a single order."""
    factor = EMISSION_FACTORS.get(mode, EMISSION_FACTORS["motorcycle"])
    return distance_km * factor


def compute_fleet_co2(df: pd.DataFrame, mode: str = "motorcycle") -> dict:
    """Compute total fleet CO2 emissions across all orders."""
    co2_series = df["distance_km"].apply(lambda d: estimate_co2_per_order(d, mode))
    total_co2_g = co2_series.sum()
    return {
        "total_co2_kg": round(total_co2_g / 1000, 2),
        "avg_co2_per_order_g": round(co2_series.mean(), 2),
        "total_distance_km": round(df["distance_km"].sum(), 2),
    }


def co2_scenario(df: pd.DataFrame, pct_electric: float) -> dict:
    """Simulate switching a percentage of deliveries to electric scooters."""
    n = len(df)
    n_electric = int(n * pct_electric / 100)
    n_motorcycle = n - n_electric

    distances = df["distance_km"].values
    np.random.seed(42)
    electric_mask = np.zeros(n, dtype=bool)
    electric_idx = np.random.choice(n, size=n_electric, replace=False)
    electric_mask[electric_idx] = True

    baseline_co2 = distances.sum() * EMISSION_FACTORS["motorcycle"] / 1000
    scenario_co2 = (
        distances[electric_mask].sum() * EMISSION_FACTORS["electric_scooter"]
        + distances[~electric_mask].sum() * EMISSION_FACTORS["motorcycle"]
    ) / 1000

    savings = baseline_co2 - scenario_co2
    savings_pct = (savings / baseline_co2 * 100) if baseline_co2 > 0 else 0.0

    return {
        "baseline_co2_kg": round(baseline_co2, 2),
        "scenario_co2_kg": round(scenario_co2, 2),
        "savings_kg": round(savings, 2),
        "savings_pct": round(savings_pct, 2),
    }


def co2_by_area(df: pd.DataFrame) -> pd.DataFrame:
    """Compute total and average CO2 per delivery area."""
    df_co2 = df.copy()
    df_co2["co2_g"] = df_co2["distance_km"] * EMISSION_FACTORS["motorcycle"]

    area_co2 = df_co2.groupby("area").agg(
        total_co2_kg=("co2_g", lambda x: round(x.sum() / 1000, 2)),
        avg_co2_per_order_g=("co2_g", lambda x: round(x.mean(), 2)),
        order_count=("order_id", "count"),
    ).reset_index()

    area_co2 = area_co2.sort_values("total_co2_kg", ascending=False).reset_index(drop=True)
    return area_co2
