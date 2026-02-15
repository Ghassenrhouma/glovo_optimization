# Glovo Delivery Optimization

An end-to-end delivery optimization platform built with Python, scikit-learn, XGBoost, LightGBM, and Streamlit. The project analyses 50,000 Glovo delivery orders to uncover operational inefficiencies, predict delays, optimise courier staffing, estimate environmental impact, and surface customer feedback insights — all through an interactive six-page dashboard.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Setup & Quick Start](#setup--quick-start)
3. [Dataset](#dataset)
4. [Source Modules (`src/`)](#source-modules-src)
   - [preprocessing.py](#1-preprocessingpy--data-loading-cleaning--feature-engineering)
   - [model.py](#2-modelpy--machine-learning-training--evaluation)
   - [clustering.py](#3-clusteringpy--area-performance-analysis)
   - [optimization.py](#4-optimizationpy--courier-load-balancing--staffing)
   - [co2.py](#5-co2py--co₂-emission-estimation)
5. [Dashboard Pages (`app/streamlit_app.py`)](#dashboard-pages-appstreamlit_apppy)
   - [Page 1 — Overview & EDA](#page-1--overview--eda)
   - [Page 2 — Area Performance](#page-2--area-performance)
   - [Page 3 — Delay Prediction](#page-3--delay-prediction)
   - [Page 4 — Courier & Staffing](#page-4--courier--staffing)
   - [Page 5 — CO₂ Impact](#page-5--co₂-impact)
   - [Page 6 — Customer Feedback](#page-6--customer-feedback)
6. [Tech Stack](#tech-stack)

---

## Project Structure

```
glovo_optimization/
├── data/
│   └── orders_details.csv          # Raw dataset (50K orders)
├── src/
│   ├── __init__.py                 # Package init
│   ├── preprocessing.py            # Data loading, cleaning, feature engineering
│   ├── model.py                    # ML training, evaluation, prediction
│   ├── clustering.py               # Area-level performance analytics
│   ├── optimization.py             # Courier load simulation & staffing
│   └── co2.py                      # CO₂ emission calculations
├── models/
│   └── best_regression_model.pkl   # Saved best model (auto-generated)
├── app/
│   └── streamlit_app.py            # Six-page Streamlit dashboard
├── notebooks/                      # Jupyter exploration notebooks
├── requirements.txt
├── README.md
└── run.sh                          # One-command launcher (Linux/Mac)
```

---

## Setup & Quick Start

### Prerequisites

- Python 3.10+
- `pip` or a virtual environment manager

### Installation

```bash
# Create and activate a virtual environment (recommended)
python -m venv venv
# Linux/Mac
source venv/bin/activate
# Windows PowerShell
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Running the Dashboard

```bash
# Ensure data/orders_details.csv is present, then:
streamlit run app/streamlit_app.py --server.port 8501
```

The app opens at `http://localhost:8501`. On first launch the ML models are automatically trained (~10-20 seconds) and the best model is saved to `models/best_regression_model.pkl`.

---

## Dataset

| Property | Value |
|----------|-------|
| Rows | 50,000 delivery orders |
| Raw Columns | 30 (order details, timing, customer info, courier, ratings) |
| Missing Values | None |
| Time Range | Multiple months of Glovo delivery data |

Key columns include: `order_id`, `order_date`, `order_total`, `distance_km`, `delivery_status` (On Time / Slightly Delayed / Significantly Delayed), `delivery_time_minutes`, `customer_segment`, `payment_method`, `rating`, `sentiment`, `feedback_category`, `area`, `delivery_partner_id`, and more.

---

## Source Modules (`src/`)

### 1. `preprocessing.py` — Data Loading, Cleaning & Feature Engineering

This module is the entry point of the data pipeline. It transforms the raw CSV into a clean, feature-rich DataFrame ready for analysis and modelling.

**Functions:**

| Function | Description |
|----------|-------------|
| `load_data(filepath)` | Reads the CSV, drops unnecessary columns (`Unnamed: 0`, `delivery_status_y`, `promised_time`, `actual_time`), renames ambiguous columns, and parses all datetime columns (`order_date`, `promised_delivery_time`, `actual_delivery_time`, `feedback_date`, `registration_date`). |
| `compute_features(df)` | Engineers new columns from the cleaned data. Creates temporal features (`hour_of_day`, `day_of_week`, `is_weekend`, `is_peak_hour`, `month`), delay flags (`is_delayed`, `is_significantly_delayed`), `customer_lifetime_days` (days since registration), `profit_margin` (profit / order total), and `order_size_category` (Small < $500, Medium < $2000, Large). |
| `encode_categoricals(df)` | Label-encodes `payment_method`, `customer_segment`, `feedback_category`, and `sentiment` into `*_encoded` columns. Maps `delivery_status` to numeric values (On Time → 0, Slightly Delayed → 1, Significantly Delayed → 2). |
| `get_feature_matrix(df)` | Returns `(X, y)` for **regression**: X contains 14 feature columns, y is `delivery_time_minutes`. |
| `get_classification_matrix(df)` | Returns `(X, y)` for **binary classification**: X contains the same 14 features, y is `is_delayed` (0 or 1). |

**Feature Columns used for modelling (14 total):**
`order_total`, `number_product`, `total_quantity`, `distance_km`, `hour_of_day`, `day_of_week`, `is_weekend`, `is_peak_hour`, `month`, `payment_method_encoded`, `customer_segment_encoded`, `total_orders`, `avg_order_value`, `profit_margin`.

---

### 2. `model.py` — Machine Learning Training & Evaluation

This module handles the full ML lifecycle: training, evaluation, model selection, persistence, and prediction.

**Functions:**

| Function | Description |
|----------|-------------|
| `train_regression_models(X_train, y_train)` | Trains three regression models — **Random Forest** (100 trees), **XGBoost**, and **LightGBM** — all with `random_state=42` for reproducibility. Returns a dictionary `{name: fitted_model}`. |
| `train_classification_models(X_train, y_train)` | Same as above but for binary classification (delay yes/no). Uses `RandomForestClassifier`, `XGBClassifier`, and `LGBMClassifier`. |
| `evaluate_regression(models, X_test, y_test)` | Computes **RMSE**, **MAE**, and **R²** for each regression model on the test set. Returns a DataFrame indexed by model name. |
| `evaluate_classification(models, X_test, y_test)` | Computes **Accuracy**, **F1-score**, and **ROC-AUC** for each classification model. |
| `save_best_model(models, metrics, path, metric_col)` | Selects the best model by the given metric (lowest RMSE/MAE or highest R²/Accuracy) and saves it to disk using `joblib`. Returns the winning model name. |
| `get_feature_importance(model, feature_names)` | Extracts `.feature_importances_` from the model and returns a sorted DataFrame showing which features contribute most to predictions. |
| `predict_delay(model, input_dict)` | Takes a dictionary of feature values, converts it to a DataFrame, runs `model.predict()`, and returns the predicted delivery delay in minutes. Used for real-time predictions in the dashboard. |

**How model selection works:**
The three models are trained on an 80/20 train-test split. The model with the lowest RMSE is automatically selected as the best, saved to `models/best_regression_model.pkl`, and used for all subsequent predictions. Training happens once at app startup via `@st.cache_resource` and is cached for the session.

---

### 3. `clustering.py` — Area Performance Analysis

This module groups orders by delivery area and computes aggregate performance metrics to identify problematic zones.

**Functions:**

| Function | Description |
|----------|-------------|
| `compute_area_stats(df)` | Groups by `area` and computes: `order_count`, `avg_delivery_delay` (mean minutes), `delay_rate_pct` (% of late orders), `significant_delay_rate_pct` (% of significantly delayed), `avg_distance_km`, `avg_order_total`, `avg_rating`, `positive_sentiment_pct`. Sorted by delay rate descending. |
| `classify_area_performance(area_stats)` | Assigns a **performance tier** to each area: 🔴 **High Risk** (delay rate > 40%), 🟡 **Medium Risk** (20–40%), 🟢 **Low Risk** (< 20%). |
| `get_top_bottom_areas(area_stats, n)` | Returns the best and worst `n` areas by delay rate as two separate DataFrames. |
| `compute_hourly_patterns(df)` | Computes average delay and delay rate per hour of day across all areas. |
| `compute_segment_analysis(df)` | Computes average order value, rating, delay rate, and order count per customer segment (New, Regular, Premium, Inactive). |

---

### 4. `optimization.py` — Courier Load Balancing & Staffing

This module simulates courier workload distribution and recommends staffing adjustments to reduce delays during peak hours.

**Functions:**

| Function | Description |
|----------|-------------|
| `simulate_courier_load(df)` | Groups by `delivery_partner_id` and computes: total deliveries, average delay (minutes), on-time rate (%), and average distance. Sorted by total deliveries descending. |
| `identify_overloaded_couriers(stats, threshold)` | Filters couriers whose total deliveries exceed the threshold (default: 20). |
| `simulate_load_redistribution(df, threshold)` | Simulates reassigning 20% of orders from overloaded couriers to underloaded ones. Assumes a 15% delay improvement per redistributed batch. Returns before/after average delay and improvement percentage. |
| `compute_peak_hour_analysis(df)` | Computes order count, average delay, unique couriers, and **utilization** (orders per courier) for each hour of the day. |
| `recommend_staffing(peak_df)` | Identifies the 5 most stressed hours using a **stress score** = utilization × average delay. Recommends extra couriers needed (assuming ≤ 10 orders per courier per hour) and estimates the expected delay reduction in minutes. |

**Redistribution logic:**
- Orders from couriers with > 20 deliveries are partially reassigned (20% of their volume).
- The simulation assumes each redistributed batch yields a 15% delay improvement.
- This is a heuristic simulation — real-world redistribution would require route-level optimisation.

---

### 5. `co2.py` — CO₂ Emission Estimation

This module estimates the carbon footprint of the delivery fleet based on distance travelled and vehicle type, and simulates electrification scenarios.

**Emission Factors (gCO₂ per km):**

| Vehicle | gCO₂/km |
|---------|---------|
| Motorcycle | 103 |
| Car | 171 |
| Electric Scooter | 22 |
| Bicycle | 0 |

**Functions:**

| Function | Description |
|----------|-------------|
| `estimate_co2_per_order(distance_km, mode)` | Returns CO₂ emissions in grams for a single delivery. Default mode is motorcycle. |
| `compute_fleet_co2(df, mode)` | Computes total fleet emissions (kg), average per order (g), and total distance (km) assuming all deliveries use the specified mode. |
| `co2_scenario(df, pct_electric)` | Simulates switching a given percentage of the fleet from motorcycles to electric scooters. Randomly selects which orders are "electric" (seeded for reproducibility). Returns baseline CO₂, scenario CO₂, savings in kg, and savings percentage. |
| `co2_by_area(df)` | Computes total and average CO₂ per delivery area (assuming motorcycle). Sorted by total emissions descending. |

---

## Dashboard Pages (`app/streamlit_app.py`)

The Streamlit dashboard is a single-file application with 6 pages accessible via a sidebar radio navigation. Data is loaded and cached with `@st.cache_data`, and the ML model is trained once at startup with `@st.cache_resource`.

### Page 1 — Overview & EDA

**Purpose:** Provide a high-level snapshot of the entire delivery dataset and uncover basic trends.

**Contents:**
- **KPI Row 1 (4 metrics):** Total Orders, Average Delay (minutes), On-Time Rate (%), Average Rating.
- **KPI Row 2 (4 metrics):** Total Revenue ($), Average Order Value ($), Average Distance (km), Unique Customers.
- **Left Column Charts:**
  - *Delivery Status Distribution* — Pie chart showing the proportion of On Time, Slightly Delayed, and Significantly Delayed orders.
  - *On-Time Rate by Customer Segment* — Bar chart comparing on-time performance across New, Regular, Premium, and Inactive customers.
  - *Distribution of Delivery Time* — Histogram of `delivery_time_minutes` with a red dashed line at 0 (on-time threshold).
- **Right Column Charts:**
  - *Average Delay by Hour of Day* — Bar chart revealing which hours have the highest average delay.
  - *Order Count by Payment Method & Sentiment* — Grouped bar chart showing order volume by payment type, coloured by customer sentiment.
  - *Daily Order Volume* — Line chart showing order trends over time.

---

### Page 2 — Area Performance

**Purpose:** Identify which delivery areas are performing well and which are problematic, enabling targeted operational improvements.

**Contents:**
- **Area Summary Table** — Full table of all areas with order count, delay rate, average distance, average rating, sentiment percentage, and a performance tier indicator (🔴 High Risk, 🟡 Medium Risk, 🟢 Low Risk).
- **Top 15 Highest Delay Rate Areas** — Red-scaled bar chart showing the worst-performing areas.
- **Top 15 Lowest Delay Rate Areas** — Green-scaled bar chart showing the best-performing areas.
- **Avg Rating — Top 20 Areas by Order Count** — Bar chart of average customer rating for the 20 busiest areas.
- **Avg Distance vs Avg Delay Scatter** — Bubble chart where each area is a point, sized by order volume and coloured by risk tier, revealing whether distance correlates with delay.

---

### Page 3 — Delay Prediction

**Purpose:** Train ML models to predict delivery delay and provide a live prediction interface.

**Contents:**
- **Model Performance Section:**
  - Shows which model was selected as best (auto-trained at startup).
  - Displays RMSE, MAE, and R² for all three models (Random Forest, XGBoost, LightGBM) in a comparison table.
  - *Top 10 Feature Importances* — Horizontal bar chart showing which features most influence delay prediction.
- **Live Delay Prediction Section:**
  - Interactive input form: Order Total, Distance, Hour of Day, Number of Products, Customer Segment, Payment Method, Weekend flag, Peak Hour flag.
  - **Predict Delay** button that runs the best model in real time and displays the predicted delay in minutes with a colour-coded status:
    - 🎉 Early delivery (negative delay)
    - ✅ On time (≤ 5 min)
    - ⚠️ Slight delay (5–15 min)
    - 🚨 Significant delay (> 15 min)

---

### Page 4 — Courier & Staffing

**Purpose:** Analyse courier workloads, simulate load redistribution, and recommend staffing levels for peak hours.

**Contents:**
- **Courier Performance Table** — Top 50 couriers by total deliveries, showing average delay, on-time rate, and average distance. Overloaded couriers (> 20 deliveries) are highlighted in red.
- **Top 20 Couriers by Avg Delay** — Bar chart of the worst-performing couriers by average delay.
- **Load Redistribution Simulation** — Displays before/after average delay and improvement percentage if 20% of overloaded orders were redistributed. Shows the number of orders that would be reassigned.
- **Peak Hour Staffing Analysis:**
  - *Dual-axis chart* — Order volume (bars) and average delay (line) by hour of day, revealing when the system is most stressed.
  - *Staffing Recommendations Table* — For the 5 most stressed hours, shows current couriers, recommended additional couriers, and expected delay reduction in minutes.

---

### Page 5 — CO₂ Impact

**Purpose:** Estimate the fleet's carbon footprint and simulate the environmental benefit of transitioning to electric vehicles.

**Contents:**
- **Fleet KPIs (3 metrics):** Total CO₂ (kg), Average CO₂ per Order (g), Total Distance (km).
- **Top 20 Areas by Total CO₂** — Bar chart showing which areas contribute most to emissions.
- **Electrification Scenario Simulator:**
  - Interactive slider to set the percentage of fleet switched to electric scooters (0–100%).
  - Before/after CO₂ metrics with savings percentage.
  - Success message showing absolute savings in kg.
  - *Fleet CO₂ vs Electrification Rate* — Line chart plotting total emissions at every 5% electrification step, with a dotted reference line at the currently selected percentage.

---

### Page 6 — Customer Feedback

**Purpose:** Analyse customer satisfaction patterns across segments, areas, and time.

**Contents:**
- **Left Column:**
  - *Sentiment Distribution* — Bar chart showing counts of Positive, Neutral, and Negative feedback.
  - *Sentiment by Customer Segment* — Grouped bar chart breaking down sentiment by New, Regular, Premium, and Inactive customers.
  - *Top 10 Lowest-Rated Areas (≥ 50 orders)* — Table of areas with the worst average rating (filtered to areas with sufficient data).
- **Right Column:**
  - *Avg Rating by Feedback Category* — Bar chart showing how different feedback topics (food quality, delivery speed, etc.) correlate with ratings.
  - *Monthly Average Rating Trend* — Line chart tracking customer satisfaction over time.
  - *Order Total vs Rating* — Scatter plot (5,000 sampled points) coloured by delivery status, showing whether higher-value orders receive better/worse ratings.

---

## Tech Stack

| Component | Libraries |
|-----------|-----------|
| Data Processing | pandas, numpy |
| Machine Learning | scikit-learn, XGBoost, LightGBM |
| Visualisation | Plotly, matplotlib, seaborn |
| Dashboard | Streamlit |
| Model Persistence | joblib |
