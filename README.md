# LSTM Power Forecasting

Project that applies a **Long Short‑Term Memory (LSTM)** neural network to forecast **daily household power consumption**. The entire workflow lives in a single Jupyter notebook and covers: **data preparation**, **feature scaling**, **sequence framing**, **modeling in PyTorch**, and **business‑oriented evaluation**.

---

## Business Context

Accurate short‑term power forecasts help with:
- **Grid reliability:** balancing supply and demand to reduce outages and costly reserve usage.
- **Cost management:** optimizing purchasing/scheduling decisions for utilities and large energy users.
- **Demand response & sustainability:** informing peak‑shaving, storage, and renewable integration.

This notebook demonstrates a repeatable deep‑learning pipeline for **next‑day power forecasting** from historical consumption patterns.

---

## Problem Statement

**Goal:** Predict the **next day’s average “Global Active Power”** using the previous **30 days** of aggregated daily signals from the household power dataset.

**Why it matters:** Even small accuracy gains translate to reduced imbalance penalties, better hedging, and improved planning decisions across utility operations and large energy consumers.

---

## Data

- **Source:** UCI Machine Learning Repository — *Individual household electric power consumption*.
- **Granularity:** Original readings at minute‑level; the notebook **resamples to daily means**.
- **Target:** `Global_active_power` (daily average).
- **Features:** All available numerical features from the dataset (e.g., Global_active_power, Global_reactive_power, Voltage, Global_intensity, Sub_metering_1–3), aggregated to daily level.

---

## Methodology (Inside the Notebook)

1. **Ingest & Clean**
   - Parse timestamps, set a DateTime index.
   - Drop rows with missing values encoded as `?` (per dataset conventions).

2. **Aggregate to Daily**
   - Resample to **daily mean** to stabilize noise and align with a day‑ahead forecasting horizon.

3. **Train/Validation Split**
   - **Training:** up to **2009‑12‑31**.
   - **Validation:** **2010‑01‑01** to **2010‑11‑26**.
   - (Dates align with common splits used in the dataset to test generalization over time.)

4. **Normalize**
   - **Min‑Max scaling** (`sklearn`) fit on the **training period** only.
   - Convert to arrays/DataFrames for convenience.

5. **Sequence Framing**
   - Build supervised sequences with a **lookback window of 30 days**:  
     `X[t] = [days t‑30..t‑1] → y[t] = day t (next‑day Global_active_power)`.

6. **Modeling — PyTorch LSTM**
   - **Architecture:** 2‑layer **LSTM** (hidden size **50**, **dropout 0.2**) → **linear head** to 1 value.
   - **Loss:** MSE; **Optimizer:** Adam (lr **1e‑3**); **Batch size:** 32; **Epochs:** **20**.
   - **Stability:** gradient clipping (`max_norm=5`).

7. **Evaluation**
   - Invert scaling for the target and compute **MSE**, **RMSE**, **MAE**, **R²** on the validation set.
   - Plot **training vs. validation loss** and **Actual vs. Predicted** (first 100 days) for stakeholder‑friendly review.

---

## Results (Qualitative Summary)

- The LSTM learns key **temporal patterns** in daily consumption, capturing **level/seasonality** present in household demand.
- Validation curves show **converging training/validation loss**, indicating reasonable generalization for a compact model.
- The sample plot (first 100 validation days) illustrates **forecast alignment** with observed daily averages—useful for operational planning.

> The notebook prints exact **MSE / RMSE / MAE / R²** values after training. Use these to compare variants (e.g., different lookbacks, hidden sizes, GRU vs. LSTM).

---

## Repository Contents

- **`LSTM_Power.ipynb`** — the single, end‑to‑end notebook (data prep → sequences → LSTM → evaluation).
- **`README.md`** — this documentation.

---

## Interpreting Outputs (for Stakeholders)

- **Loss curves:** Indicate learning progress and whether the model is over/under‑fitting.
- **Actual vs. Predicted plot:** Quick visual check of forecast quality during the validation window.
- **Error metrics (RMSE/MAE/R²):** Quantify accuracy and enable comparisons across model variants.

---

## Limitations & Next Steps

- **Feature context:** The notebook uses only intrinsic consumption signals. Incorporating **exogenous drivers** (weather, holidays, day‑of‑week) typically improves accuracy.
- **Horizon:** Demonstrates **next‑day** forecasting; extend to **multi‑step** or **probabilistic** forecasts as needed.
- **Tuning:** Explore **lookback**, **hidden sizes**, **layers**, **dropout**, **learning rate**, and **optimizers**. Consider **GRU** or **CNN‑LSTM** baselines.
- **Operationalization:** Export a lightweight inference function and add thresholding/business rules for production use.
