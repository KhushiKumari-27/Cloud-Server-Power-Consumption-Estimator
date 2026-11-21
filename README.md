# ⚡ Cloud Infrastructure Power Consumption Estimator

An end-to-end Machine Learning regression pipeline designed to forecast data center server electricity consumption (in kWh) ahead of time—enabling proactive energy budgeting and data-driven hardware optimization.

---

## 📌 Business Problem
Cloud data centers spend millions of dollars annually on electricity, yet Finance and DevOps teams often discover their exact power usage only *after* the monthly utility bill arrives. Without accurate forecasting, companies struggle to budget operational costs or identify energy-inefficient legacy hardware.

## 💡 Solution
This project builds a supervised **Linear Regression** model trained on 1,500+ raw server telemetry logs (`CPU utilization`, `memory usage`, `network traffic`, `hardware generation`, and `timestamps`). By predicting power consumption across planned workloads, teams can accurately estimate monthly electricity demand before billing cycles end.

---

## 🛠️ End-to-End ML Workflow

1. **Exploratory Data Analysis (EDA):** Inspected feature distributions, missing values, and statistical anomalies across server logs.
2. **Anomaly Detection & Cleaning:** 
   * Applied the **Interquartile Range (IQR)** method to remove faulty sensor power spikes while preserving valid exponential network traffic bursts.
   * Handled missing memory sensor logs using **median imputation**.
3. **Feature Engineering & Encoding:**
   * Applied **manual ordinal encoding** (`Legacy: 3`, `Current: 2`, `Next-Gen: 1`) to capture the energy efficiency hierarchy across hardware generations.
   * Engineered binary temporal indicators (`is_weekend`, `is_peak_hours`) to avoid cyclical linear distortion, alongside a composite `avg_system_load` metric.
4. **Leakage-Free Preprocessing:** Enforced an 80/20 `train_test_split` *prior* to fitting `StandardScaler` to prevent data leakage.
5. **Model Training & Validation:** Trained a Scikit-Learn `LinearRegression` model and evaluated generalization across training and unseen test splits.

---

## 📊 Key Results & Performance

| Metric | Training Set | Testing Set | Business Interpretation |
| :--- | :---: | :---: | :--- |
| **R² Score** | — | **0.9182** | Explains **91.8%** of the variance in server power usage. |
| **MAE** | 16.13 kWh | **15.88 kWh** | Average prediction error per individual 90-minute log. |
| **RMSE** | 22.45 kWh | **22.45 kWh** | Identical train/test error confirms **zero overfitting**. |
| **MSE** | 504.13 | **504.09** | Minimized squared error across unseen server workloads. |

* **Aggregate Forecasting Impact:** Across the test cycle, positive and negative individual residuals balanced out to deliver **99.86% accuracy** on total energy demand (**58,756.58 kWh predicted** vs. **58,836.33 kWh actual**).

---

## 🧰 Tech Stack
* **Language:** Python 3
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (`LinearRegression`, `StandardScaler`, `train_test_split`)
