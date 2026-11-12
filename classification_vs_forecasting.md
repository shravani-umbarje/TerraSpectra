# Classification vs. Forecasting: A Scientific Distinction

**Author:** Shravani (ML Specialist)  
**Day:** 1  
**Purpose:** Clarify what the Tomato Bacterial Wilt dataset can and cannot support, and how this project will handle the distinction honestly.

---

## 1. Three Related but Distinct Tasks

When applying machine learning to plant disease spectral data, three tasks are often conflated. They differ fundamentally in what question they answer and what data they require.

### 1.1 Disease Classification (Diagnosis)

**Question:** *"Given a spectral measurement taken right now, what is the current disease severity?"*

- **Input:** A single spectral observation (one time point).
- **Output:** A categorical severity label (e.g., Healthy, Mild, Moderate, Severe, Critical).
- **Data requirement:** Cross-sectional data — measurements from multiple plants at various severity stages, all collected at one or a few time points.
- **ML framing:** Standard supervised multi-class classification.
- **Analogy:** A doctor diagnosing a patient's current condition from today's blood test.

### 1.2 Early Detection (Pre-symptomatic Identification)

**Question:** *"Is this plant infected even though it doesn't show visible symptoms yet?"*

- **Input:** A spectral observation from an apparently healthy plant.
- **Output:** Binary — "pre-symptomatic infection detected" vs. "truly healthy."
- **Data requirement:** Spectral measurements of plants that are confirmed to be infected (e.g., via pathogen assay) but not yet visually symptomatic, alongside truly healthy controls.
- **ML framing:** Binary classification with a focus on detecting subtle spectral shifts before visible symptom onset.
- **Analogy:** An early-warning blood test that detects cancer markers before symptoms appear.

### 1.3 Disease Forecasting (Progression Prediction)

**Question:** *"Given measurements over time, what severity will this plant reach in T days?"*

- **Input:** A **time series** of spectral observations from the same plant across multiple dates.
- **Output:** Predicted future severity (regression or classification at a future time point).
- **Data requirement:** **Longitudinal (temporal) data** — repeated spectral measurements of the same plants tracked over the disease progression timeline.
- **ML framing:** Time series forecasting, sequence modelling (LSTM, temporal CNN, or autoregressive models).
- **Analogy:** Predicting a patient's prognosis from a series of weekly check-ups.

---

## 2. What the Tomato Bacterial Wilt Dataset Supports

### Dataset Structure

The Tomato Bacterial Wilt dataset consists of:
- **293 spectral samples** (plant-level mean reflectance)
- **204 spectral bands** (400–1000 nm, VNIR range)
- **5 severity classes** (representing different stages of bacterial wilt)
- **Cross-sectional design** — plants at various severity stages were measured, but there is **no temporal axis** linking measurements of the same plant over time.

### What This Means

| Task | Supported? | Reason |
|------|-----------|--------|
| **Disease classification** | ✅ Yes | The dataset provides labelled samples at different severity levels. A classifier trained on this data can predict the current severity of a new plant based on its spectrum. |
| **Early detection** | 🔶 Partially | If severity class 1 (mild/early) represents a pre-symptomatic or very early stage, the classifier implicitly performs early detection by distinguishing class 1 from class 0 (healthy). However, without pathogen confirmation independent of visual symptoms, this is not rigorous early detection. |
| **Disease forecasting** | ❌ No | There is no temporal dimension — we cannot track how a specific plant's spectrum changes over days or weeks. Each sample is a snapshot, not a time series. |

### Why "Forecasting" Would Be Scientifically Inaccurate

Calling this project's output a "disease forecast" would be misleading because:

1. **No temporal ordering.** The samples are not indexed by time. We have no information about when each measurement was taken relative to infection onset.
2. **No repeated measurements.** Each plant (or leaf) appears once. We cannot observe progression trajectories.
3. **No lead-time validation.** Forecasting requires evaluating whether a prediction made at time *t* correctly anticipates the state at time *t + Δt*. This is impossible without temporal data.
4. **Severity ≠ future state.** The 5 severity classes describe the plant's *current* condition, not where it will be tomorrow. A plant at severity 2 today will not necessarily reach severity 4 next week — it depends on treatment, environment, and pathogen dynamics.

---

## 3. What Genuine Forecasting Would Require

For a true disease forecasting system, the dataset would need:

### 3.1 Longitudinal Study Design

- **Repeated spectral measurements** of the same plants at regular intervals (e.g., daily or every 2–3 days) from healthy to fully symptomatic.
- **Plant tracking:** Each sample linked to a specific plant ID, with a time stamp.
- **Controlled inoculation:** Plants deliberately infected at a known date, with uninoculated controls.

### 3.2 Time Series Features

| Feature | Description |
|---------|-------------|
| Spectral change rate | How quickly the spectrum shifts from healthy baseline |
| Red-edge shift velocity | Rate of red-edge position change (nm/day) |
| NDVI decline slope | Rate of vegetation index decline |
| Temporal spectral derivatives | First and second derivatives of reflectance over time |

### 3.3 Modelling Approaches for Forecasting

- **LSTM / GRU networks:** Learn temporal dependencies in sequential spectral observations.
- **Temporal CNN:** 1D convolutions along the time axis (spectral features as channels).
- **Autoregressive models:** Predict next-day spectrum from previous days' spectra.
- **Cox proportional hazards:** Model the "time to event" (e.g., time to severity 4) as a survival analysis problem.

### 3.4 Minimum Data Volume

A rough estimate for viable temporal forecasting:
- ≥ 50 individual plants tracked over time
- ≥ 10 time points per plant
- = 500+ temporally linked observations minimum

---

## 4. The Pine Wilt Dataset — Potential Temporal Exploration

The Early Monitoring Dataset for Pine Wilt Disease (the second dataset in TerraSpectra) may contain temporal information that the Tomato Wilt dataset lacks.

### What to Investigate (Day 2+)

1. **Does the Pine Wilt dataset include temporal metadata?** (dates of measurement, repeated visits to same trees)
2. **Are individual trees tracked over time?** (tree IDs with multiple measurement dates)
3. **What is the measurement frequency?** (daily, weekly, monthly, seasonal?)
4. **Can a temporal progression be reconstructed?** (even if retrospectively)

If the Pine Wilt dataset does contain temporal data:
- It could support a **proof-of-concept temporal model** alongside the Tomato Wilt classification model.
- The project could honestly demonstrate both classification (Tomato) and temporal analysis (Pine), showing the difference in what they achieve.
- However, this would be a **secondary, exploratory** analysis — not the primary deliverable.

---

## 5. How This Project Will Handle the Distinction Honestly

### 5.1 Terminology

Throughout all project documentation, code comments, and the final dashboard:

| ✅ We WILL say | ❌ We will NOT say |
|----------------|-------------------|
| "Disease severity classification" | "Disease prediction" (ambiguous) |
| "Current severity assessment" | "Disease forecasting" |
| "Spectral-based diagnosis" | "Predicting future disease" |
| "Classifying wilt stage from spectrum" | "Forecasting crop health" |
| "Identifying the current severity level" | "Early warning system" (unless rigorously validated) |

### 5.2 Dashboard Labels

Abdul's dashboard will label the ML output as:

- **"Predicted Current Severity"** — not "Predicted Disease Outcome"
- **"Classification Confidence"** — not "Forecast Probability"
- **"Severity Assessment Based on Spectral Signature"** — not "Future Disease Risk"

### 5.3 Report and Documentation

The final project report will include:

1. A clear statement that the Tomato Wilt model performs **classification of current severity** based on spectral reflectance.
2. An explicit note that the dataset does not support temporal forecasting.
3. A "Future Work" section describing what data collection would be needed for genuine forecasting.
4. If Pine Wilt temporal analysis is conducted, a clearly marked section explaining it as exploratory.

### 5.4 Scientific Integrity Statement

> **This project classifies the current severity of tomato bacterial wilt from hyperspectral leaf reflectance data. It does not forecast future disease progression, as the dataset contains cross-sectional (single time point) measurements only. Any reference to "prediction" in this project refers to the ML model's prediction of the current severity class, not a temporal forecast of future plant health.**

---

## 6. Summary Table

| Dimension | Classification (Our Project) | Forecasting (Not Supported) |
|-----------|-----------------------------|-----------------------------|
| **Data type** | Cross-sectional snapshots | Longitudinal time series |
| **Input** | Single spectrum | Sequence of spectra over time |
| **Output** | Current severity label | Future severity / time to event |
| **Temporal axis** | None | Required |
| **Plant tracking** | Not required | Required |
| **ML approach** | RF, SVM, XGBoost | LSTM, temporal CNN, survival analysis |
| **Validation** | Standard CV on snapshots | Temporal train/test split |
| **Our dataset** | ✅ 293 labelled snapshots | ❌ No temporal data |
| **Honest claim** | "Classifies current severity" | Would need new data collection |
