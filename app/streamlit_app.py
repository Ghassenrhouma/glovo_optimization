"""Glovo Delivery Optimization — Streamlit Dashboard."""

import sys
import os

# Add parent directory so src/ imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from sklearn.model_selection import train_test_split

from src.preprocessing import (
    load_data,
    compute_features,
    encode_categoricals,
    get_feature_matrix,
    get_classification_matrix,
    FEATURE_COLS,
)
from src.model import (
    train_regression_models,
    evaluate_regression,
    save_best_model,
    get_feature_importance,
    predict_delay,
)
from src.clustering import (
    compute_area_stats,
    classify_area_performance,
    get_top_bottom_areas,
    compute_hourly_patterns,
    compute_segment_analysis,
)
from src.optimization import (
    compute_zone_performance,
    identify_overloaded_zones,
    simulate_load_redistribution,
    compute_peak_hour_analysis,
    recommend_staffing,
)
from src.co2 import (
    compute_fleet_co2,
    co2_scenario,
    co2_by_area,
)

# ─── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Glovo Delivery Optimization", page_icon="🛵", layout="wide")

# ─── Data loading ──────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "orders_details.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


@st.cache_data
def load_and_prepare():
    """Load, clean, feature-engineer and encode the full dataset."""
    df = load_data(DATA_PATH)
    df = compute_features(df)
    df = encode_categoricals(df)
    return df


@st.cache_resource
def train_and_save_model(_df):
    """Train regression models at startup and save the best one."""
    X, y = get_feature_matrix(_df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    models = train_regression_models(X_train, y_train)
    metrics = evaluate_regression(models, X_test, y_test)

    model_path = os.path.join(MODEL_DIR, "best_regression_model.pkl")
    best_name = save_best_model(models, metrics, model_path, metric_col="RMSE")
    best_model = models[best_name]
    fi = get_feature_importance(best_model, list(X.columns))

    return best_model, best_name, metrics, fi


df = load_and_prepare()
best_model, best_model_name, reg_metrics, feature_importances = train_and_save_model(df)

# ─── Sidebar ───────────────────────────────────────────────────────────────────
page = st.sidebar.radio(
    "Navigation",
    [
        "📊 Overview & EDA",
        "🗺️ Area Performance",
        "🤖 Delay Prediction",
        "🚚 Courier & Staffing",
        "🌱 CO2 Impact",
        "💬 Customer Feedback",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Dataset:** {df.shape[0]:,} rows × {df.shape[1]} cols")
st.sidebar.markdown(f"**Unique customers:** {df['customer_id'].nunique():,}")
st.sidebar.markdown(f"**Last order:** {df['order_date'].max().strftime('%Y-%m-%d')}")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Overview & EDA
# ═══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview & EDA":
    st.title("📊 Overview & Exploratory Data Analysis")

    # KPI row 1
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Orders", f"{len(df):,}")
    c2.metric("Avg Delay (min)", f"{df['delivery_time_minutes'].mean():.2f}")
    on_time_rate = (df["is_delayed"] == 0).mean() * 100
    c3.metric("On-Time Rate", f"{on_time_rate:.1f}%")
    c4.metric("Avg Rating", f"{df['rating'].mean():.2f}")

    # KPI row 2
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Total Revenue", f"${df['order_total'].sum():,.0f}")
    c6.metric("Avg Order Value", f"${df['order_total'].mean():.2f}")
    c7.metric("Avg Distance (km)", f"{df['distance_km'].mean():.2f}")
    c8.metric("Unique Customers", f"{df['customer_id'].nunique():,}")

    st.markdown("---")

    # Charts
    col_left, col_right = st.columns(2)

    with col_left:
        # Pie chart: delivery status distribution
        status_counts = df["delivery_status"].value_counts().reset_index()
        status_counts.columns = ["status", "count"]
        fig = px.pie(status_counts, values="count", names="status",
                     title="Delivery Status Distribution",
                     color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig, use_container_width=True)

        # Bar: rating distribution (1-5 stars)
        rating_dist = df["rating"].value_counts().sort_index().reset_index()
        rating_dist.columns = ["Rating", "Count"]
        fig = px.bar(rating_dist, x="Rating", y="Count",
                     title="Rating Distribution (1–5 Stars)",
                     color="Rating", color_continuous_scale="RdYlGn")
        st.plotly_chart(fig, use_container_width=True)

        # Histogram: delivery_time_minutes
        fig = px.histogram(df, x="delivery_time_minutes", nbins=50,
                           title="Distribution of Delivery Time (minutes from promise)",
                           color_discrete_sequence=["#636EFA"])
        fig.add_vline(x=0, line_dash="dash", line_color="red",
                      annotation_text="On Time")
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        # Bar: average delay by hour
        hourly = df.groupby("hour_of_day")["delivery_time_minutes"].mean().reset_index()
        fig = px.bar(hourly, x="hour_of_day", y="delivery_time_minutes",
                     title="Average Delay by Hour of Day",
                     labels={"hour_of_day": "Hour", "delivery_time_minutes": "Avg Delay (min)"},
                     color_discrete_sequence=["#EF553B"])
        st.plotly_chart(fig, use_container_width=True)

        # Bar: order count by day of week
        dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        dow = df.groupby("day_of_week").size().reset_index(name="count")
        dow["day_name"] = dow["day_of_week"].map(dict(enumerate(dow_labels)))
        fig = px.bar(dow, x="day_name", y="count",
                     title="Order Count by Day of Week",
                     color="count", color_continuous_scale="Blues")
        st.plotly_chart(fig, use_container_width=True)

        # Line chart: daily order volume
        daily = df.set_index("order_date").resample("D").size().reset_index(name="orders")
        fig = px.line(daily, x="order_date", y="orders",
                      title="Daily Order Volume",
                      labels={"order_date": "Date", "orders": "Orders"})
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Area Performance
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🗺️ Area Performance":
    st.title("🗺️ Area Performance Analysis")

    area_stats = compute_area_stats(df)
    area_stats = classify_area_performance(area_stats)

    # Risk emoji indicator
    def risk_emoji(tier):
        if tier == "High Risk":
            return "🔴 High Risk"
        elif tier == "Medium Risk":
            return "🟡 Medium Risk"
        return "🟢 Low Risk"

    st.subheader("Area Summary Table")
    area_display = area_stats.copy()
    area_display["performance_tier"] = area_display["performance_tier"].apply(risk_emoji)
    st.dataframe(area_display, use_container_width=True, height=400)

    col_l, col_r = st.columns(2)

    with col_l:
        # Worst 15
        worst = area_stats.nlargest(15, "delay_rate_pct")
        fig = px.bar(worst, x="area", y="delay_rate_pct",
                     title="Top 15 Areas — Highest Delay Rate (%)",
                     color="delay_rate_pct", color_continuous_scale="Reds")
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        # Best 15
        best = area_stats.nsmallest(15, "delay_rate_pct")
        fig = px.bar(best, x="area", y="delay_rate_pct",
                     title="Top 15 Areas — Lowest Delay Rate (%)",
                     color="delay_rate_pct", color_continuous_scale="Greens_r")
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    # Avg rating by top 20 areas (by order count)
    top20 = area_stats.nlargest(20, "order_count")
    fig = px.bar(top20, x="area", y="avg_rating",
                 title="Avg Rating — Top 20 Areas by Order Count",
                 color="avg_rating", color_continuous_scale="Viridis")
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    # Scatter: distance vs delay
    fig = px.scatter(
        area_stats, x="avg_distance_km", y="avg_delivery_delay",
        color="performance_tier", size="order_count",
        title="Avg Distance vs Avg Delay by Area",
        color_discrete_map={"High Risk": "red", "Medium Risk": "orange", "Low Risk": "green"},
        hover_name="area",
    )
    st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Delay Prediction Model
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Delay Prediction":
    st.title("🤖 Delay Prediction Model")

    # ── Section 1: Model Performance (auto-trained at startup) ──
    st.subheader("Model Performance")
    st.success(f"Model auto-trained at startup — Best model: **{best_model_name}**")
    st.dataframe(reg_metrics.style.format(precision=4), use_container_width=True)

    fig = px.bar(
        feature_importances.head(10), x="importance", y="feature", orientation="h",
        title="Top 10 Feature Importances",
        color="importance", color_continuous_scale="Blues",
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Section 2: Live Prediction ──
    st.subheader("Live Delay Prediction")

    col1, col2 = st.columns(2)

    with col1:
        order_total = st.number_input("Order Total ($)", min_value=0.0, value=500.0, step=50.0)
        distance_km = st.slider("Distance (km)", 0.5, 5.0, 2.5, 0.1)
        hour_of_day = st.slider("Hour of Day", 0, 23, 12)
        number_product = st.slider("Number of Products", 1, 8, 3)

    with col2:
        segment_options = ["New", "Regular", "Premium", "Inactive"]
        customer_segment = st.selectbox("Customer Segment", segment_options)
        payment_options = ["Cash", "Card", "UPI", "Wallet"]
        payment_method = st.selectbox("Payment Method", payment_options)
        is_weekend = st.checkbox("Weekend")
        is_peak_hour = st.checkbox("Peak Hour")

    # Encode inputs
    seg_map = {s: i for i, s in enumerate(sorted(segment_options))}
    pay_map = {p: i for i, p in enumerate(sorted(payment_options))}

    input_dict = {
        "order_total": order_total,
        "number_product": number_product,
        "total_quantity": number_product,  # approximate
        "distance_km": distance_km,
        "hour_of_day": hour_of_day,
        "day_of_week": 5 if is_weekend else 2,  # rough mapping
        "is_weekend": int(is_weekend),
        "is_peak_hour": int(is_peak_hour),
        "month": 6,
        "payment_method_encoded": pay_map.get(payment_method, 0),
        "customer_segment_encoded": seg_map.get(customer_segment, 0),
        "total_orders": 5,
        "avg_order_value": order_total,
        "profit_margin": 0.15,
    }

    if st.button("🔮 Predict Delay"):
        pred = predict_delay(best_model, input_dict)
        st.metric("Predicted Delay (min)", f"{pred:.2f}")

        if pred < 0:
            st.success("Early delivery! 🎉")
        elif pred <= 5:
            st.info("On time ✅")
        elif pred <= 15:
            st.warning("Slight delay ⚠️")
        else:
            st.error("Significant delay 🚨")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Courier & Staffing Optimization
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🚚 Courier & Staffing":
    st.title("🚚 Courier & Staffing Optimization")

    # ── Section 1: Zone Performance ──
    st.subheader("Courier Zone Performance")
    st.caption(
        "Each area acts as a courier zone. Metrics show collective courier "
        "performance per zone (on-time rate, avg delay, orders per courier)."
    )

    zone_stats = compute_zone_performance(df)

    # Emoji tier for on-time rate
    def zone_tier(rate):
        if rate < 50:
            return "🔴 Critical"
        elif rate < 70:
            return "🟡 Warning"
        return "🟢 Good"

    zone_display = zone_stats.copy()
    zone_display["status"] = zone_display["on_time_rate_pct"].apply(zone_tier)
    st.dataframe(zone_display, use_container_width=True, height=400)

    col_l, col_r = st.columns(2)

    with col_l:
        # Worst 15 zones by on-time rate
        worst = zone_stats.nsmallest(15, "on_time_rate_pct")
        fig = px.bar(worst, x="area", y="on_time_rate_pct",
                     title="15 Worst Zones — Lowest On-Time Rate (%)",
                     color="on_time_rate_pct", color_continuous_scale="Reds_r")
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        # Top 15 zones by avg delay
        top_delay = zone_stats.nlargest(15, "avg_delay_minutes")
        fig = px.bar(top_delay, x="area", y="avg_delay_minutes",
                     title="15 Worst Zones — Highest Avg Delay (min)",
                     color="avg_delay_minutes", color_continuous_scale="Reds")
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    # Scatter: on-time rate vs avg distance
    fig = px.scatter(
        zone_stats, x="avg_distance_km", y="on_time_rate_pct",
        size="total_deliveries", color="avg_delay_minutes",
        hover_name="area",
        title="On-Time Rate vs Avg Distance by Zone",
        color_continuous_scale="RdYlGn_r",
        labels={"avg_distance_km": "Avg Distance (km)", "on_time_rate_pct": "On-Time Rate (%)"},
    )
    st.plotly_chart(fig, use_container_width=True)

    # Load redistribution
    st.subheader("Zone Load Redistribution Simulation")
    redist = simulate_load_redistribution(df)
    c1, c2, c3 = st.columns(3)
    c1.metric("Before Avg Delay", f"{redist['before_avg_delay']:.2f} min")
    c2.metric(
        "After Avg Delay",
        f"{redist['after_avg_delay']:.2f} min",
        delta=f"-{redist['improvement_pct']:.2f}%",
    )
    c3.metric("Orders Redistributed", f"{redist['orders_redistributed']:,}")
    st.info(
        f"**{redist['high_delay_zones']}** high-delay zones identified. "
        f"20% of their orders ({redist['orders_redistributed']:,}) would be "
        f"shifted to **{redist['low_delay_zones']}** lower-delay zones."
    )

    st.markdown("---")

    # ── Section 2: Peak Hour Staffing ──
    st.subheader("Peak Hour Staffing Analysis")

    peak_df = compute_peak_hour_analysis(df)

    # Dual-axis chart
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(x=peak_df["hour_of_day"], y=peak_df["order_count"],
               name="Order Count", marker_color="#636EFA"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=peak_df["hour_of_day"], y=peak_df["avg_delay_minutes"],
                   name="Avg Delay (min)", mode="lines+markers",
                   marker_color="#EF553B"),
        secondary_y=True,
    )
    fig.update_layout(title="Order Volume & Avg Delay by Hour")
    fig.update_yaxes(title_text="Order Count", secondary_y=False)
    fig.update_yaxes(title_text="Avg Delay (min)", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — CO2 Impact
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🌱 CO2 Impact":
    st.title("🌱 CO2 Impact Analysis")

    fleet = compute_fleet_co2(df)
    c1, c2, c3 = st.columns(3)
    c1.metric("Total CO₂ (kg)", f"{fleet['total_co2_kg']:,.1f}")
    c2.metric("Avg per Order (g)", f"{fleet['avg_co2_per_order_g']:.1f}")
    c3.metric("Total Distance (km)", f"{fleet['total_distance_km']:,.1f}")

    st.markdown("---")

    # CO2 by area
    area_co2 = co2_by_area(df)
    fig = px.bar(area_co2.head(20), x="area", y="total_co2_kg",
                 title="Top 20 Areas by Total CO₂ (kg)",
                 color="total_co2_kg", color_continuous_scale="YlOrRd")
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Electrification scenario ──
    st.subheader("⚡ Electrification Scenario")
    pct = st.slider("% of fleet switched to electric scooter", 0, 100, 30, 5)
    scenario = co2_scenario(df, pct)

    c1, c2 = st.columns(2)
    c1.metric("Baseline CO₂ (kg)", f"{scenario['baseline_co2_kg']:,.1f}")
    c2.metric(
        "Scenario CO₂ (kg)",
        f"{scenario['scenario_co2_kg']:,.1f}",
        delta=f"-{scenario['savings_pct']:.1f}%",
    )
    st.success(f"💡 Savings: **{scenario['savings_kg']:,.1f} kg** CO₂ ({scenario['savings_pct']:.1f}% reduction)")

    # Precomputed line chart: CO2 across all percentages
    @st.cache_data
    def precompute_co2_curve(_df):
        """Precompute CO2 savings at every 5% electrification step."""
        results = []
        for p in range(0, 101, 5):
            s = co2_scenario(_df, p)
            results.append({
                "electric_pct": p,
                "co2_kg": s["scenario_co2_kg"],
                "savings_kg": s["savings_kg"],
            })
        return pd.DataFrame(results)

    curve = precompute_co2_curve(df)
    fig = px.line(curve, x="electric_pct", y="co2_kg",
                  title="Fleet CO₂ vs Electrification Rate",
                  labels={"electric_pct": "% Electric Scooters", "co2_kg": "Total CO₂ (kg)"},
                  markers=True)
    fig.add_hline(y=scenario["scenario_co2_kg"], line_dash="dot", line_color="green",
                  annotation_text=f"Current: {pct}%")
    st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — Customer Feedback
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "💬 Customer Feedback":
    st.title("💬 Customer Feedback Analysis")

    col_l, col_r = st.columns(2)

    with col_l:
        # Rating distribution
        rating_dist = df["rating"].value_counts().sort_index().reset_index()
        rating_dist.columns = ["Stars", "Count"]
        fig = px.bar(rating_dist, x="Stars", y="Count",
                     title="Customer Rating Distribution",
                     color="Stars", color_continuous_scale="RdYlGn")
        st.plotly_chart(fig, use_container_width=True)

        # Avg rating by delivery status — strong signal
        status_rating = df.groupby("delivery_status")["rating"].mean().reset_index()
        status_rating.columns = ["Delivery Status", "Avg Rating"]
        fig = px.bar(status_rating, x="Delivery Status", y="Avg Rating",
                     title="Avg Rating by Delivery Status",
                     color="Avg Rating", color_continuous_scale="RdYlGn")
        fig.update_yaxes(range=[0, 5])
        st.plotly_chart(fig, use_container_width=True)

        # Top 10 lowest-rated areas (min 50 orders)
        area_ratings = df.groupby("area").agg(
            avg_rating=("rating", "mean"),
            order_count=("order_id", "count"),
        ).reset_index()
        area_ratings = area_ratings[area_ratings["order_count"] >= 50]
        worst_rated = area_ratings.nsmallest(10, "avg_rating")
        st.subheader("Top 10 Lowest-Rated Areas (≥50 orders)")
        st.dataframe(worst_rated.reset_index(drop=True), use_container_width=True)

    with col_r:
        # Rating distribution by delivery status
        rat_status = df.groupby(["delivery_status", "rating"]).size().reset_index(name="count")
        fig = px.bar(rat_status, x="rating", y="count", color="delivery_status",
                     title="Rating Breakdown by Delivery Status",
                     barmode="group",
                     labels={"rating": "Rating (Stars)", "count": "Order Count"})
        st.plotly_chart(fig, use_container_width=True)

        # Box plot: delivery time distribution by rating
        fig = px.box(df, x="rating", y="delivery_time_minutes",
                     title="Delivery Time Distribution by Rating",
                     color="rating", color_discrete_sequence=px.colors.sequential.RdBu,
                     labels={"rating": "Rating (Stars)", "delivery_time_minutes": "Delay (min)"})
        st.plotly_chart(fig, use_container_width=True)

        # Top 15 areas by % low ratings (1-2 stars)
        area_low = df.copy()
        area_low["is_low_rating"] = (area_low["rating"] <= 2).astype(int)
        area_low_pct = area_low.groupby("area").agg(
            low_rating_pct=("is_low_rating", lambda x: round(x.mean() * 100, 1)),
            order_count=("order_id", "count"),
        ).reset_index()
        area_low_pct = area_low_pct[area_low_pct["order_count"] >= 50]
        worst_low = area_low_pct.nlargest(15, "low_rating_pct")
        fig = px.bar(worst_low, x="area", y="low_rating_pct",
                     title="Top 15 Areas — Highest % of Low Ratings (1-2 Stars, ≥50 orders)",
                     color="low_rating_pct", color_continuous_scale="Reds")
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
