# 🏎️ F1 Race Strategy Optimizer

> ML-powered race strategy simulation system trained on 2024, 2025 & 2026 F1 seasons.

![Python](https://img.shields.io/badge/Python-3.13-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-R²%3D0.9955-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![Data](https://img.shields.io/badge/Data-47k%2B%20laps-orange)

---

## 🎯 What It Does

This system predicts F1 lap times and optimizes race strategies using real telemetry data from the FastF1 API. It answers questions like:

- **"What is the optimal pit stop lap for Verstappen at Bahrain?"**
- **"Is a 1-stop MEDIUM→HARD faster than a 2-stop strategy?"**
- **"How does the 2026 regulation change affect lap time predictions?"**

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| R² Score | **0.9955** |
| MAE | **0.422s** per lap |
| RMSE | **0.680s** |
| Model | XGBoost |
| Training Data | 47,000+ laps |

---

## 🏗️ Architecture
```
FastF1 API (2024 + 2025 + 2026)
        ↓
Data Pipeline (cleaning, filtering)
        ↓
Feature Engineering
  ├── CircuitBaseline    (circuit normalization)
  ├── TyreLife + TyreLife²  (degradation)
  ├── CompoundEncoded    (SOFT/MEDIUM/HARD)
  ├── CompoundTireInteraction
  ├── IsNewTire
  ├── StintProgress
  ├── DriverEncoded
  ├── FuelEffect
  ├── RegulationEra      (🆕 2026 new regs flag)
  └── Position, LapNumber, RaceEncoded
        ↓
XGBoost Regressor (500 trees, depth 8)
        ↓
Strategy Simulator + Streamlit Dashboard
```

---

## 🚀 Features

### ⚡ Quick Prediction
Predict a single lap time given driver, compound, tire age, and circuit.

### 🏁 Strategy Comparison
Compare up to 5 custom strategies side by side with lap time evolution charts.

### 🔍 Pit Window Optimizer
Find the optimal pit stop lap for any 1-stop strategy on any circuit.

### 📈 Model Performance
Full breakdown of model metrics, feature importance, and dataset statistics.

---

## 🔬 Key Technical Decisions

**1. Circuit Normalization**
Instead of predicting absolute lap times (which vary from 75s at Monaco to 96s at Bahrain), the model predicts `LapTimeDelta = LapTime - CircuitBaseline`. This reduced prediction error by 85%.

**2. Regulation Era Feature**
The 2026 F1 regulations introduced new hybrid powertrains, making cars ~8s slower. Rather than training a separate model, a `RegulationEra` flag (0=2024-25, 1=2026) allows the single model to handle both eras while maintaining R²=0.9955.

**3. Non-linear Degradation**
Tire degradation is non-linear — `TyreLife²` captures the accelerating degradation in the final laps of a stint, improving accuracy on older tires.

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Data Source | FastF1 API (Ergast) |
| ML Model | XGBoost |
| Feature Engineering | Pandas, NumPy, Scikit-learn |
| Dashboard | Streamlit |
| Visualization | Plotly |
| Deployment | Streamlit Cloud |

---

## 📁 Project Structure
```
f1-strategy-optimizer/
├── app.py                          # Streamlit dashboard
├── requirements.txt
├── data/
│   └── all_laps_2024_2025_2026.csv # 47k+ laps, 3 seasons
├── models/
│   ├── xgboost_v2_with2026.pkl     # Trained XGBoost model
│   ├── model_metadata_v2.json      # Model config & metrics
│   ├── le_driver_v2.pkl            # Driver encoder
│   └── le_race_v2.pkl              # Circuit encoder
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_lap_time_model.ipynb
│   ├── 03_strategy_simulator.ipynb
│   ├── 04_advanced_model.ipynb
│   └── 05_strategy_simulator_v2.ipynb
└── outputs/
    ├── model_comparison.png
    └── strategy_analysis_bahrain2025.png
```

---

## ⚙️ Local Setup
```bash
git clone https://github.com/chahednefzi/f1-strategy-optimizer.git
cd f1-strategy-optimizer

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
streamlit run app.py
```

---

## 📈 Results

**Bahrain GP 2025 — Verstappen Strategy Analysis:**

| Strategy | Total Time | Delta |
|----------|-----------|-------|
| 1-stop MEDIUM→HARD (lap 28) | 84.85 min | — |
| 1-stop MEDIUM→HARD (lap 20) | 84.93 min | +4.7s |
| 1-stop SOFT→HARD | 85.40 min | +32.9s |
| 2-stops SOFT→MEDIUM→HARD | 86.45 min | +96.1s |

**Key Insight:** 1-stop strategies are ~71 seconds faster than 2-stops at Bahrain. Optimal pit window: laps 26–31.

---

## 🧠 What I Learned

- **Domain expertise matters** — without understanding F1 tire strategy, the model features would have been wrong
- **Circuit normalization is critical** — predicting delta instead of absolute lap time was the key breakthrough (R² from 0.18 → 0.9955)
- **Regulation changes need explicit modeling** — the `RegulationEra` flag cleanly handles the 2026 rule changes without retraining a separate model

---

## 📬 Contact

**Chahed Nefzi** — [GitHub](https://github.com/chahednefzi)

---

*Data sourced from FastF1 API. This project is for educational and portfolio purposes.*