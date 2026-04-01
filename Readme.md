# 🏎️ F1 Race Strategy Optimizer  
### Machine Learning–Driven Race Simulation & Decision Support System

---

## 📌 Overview

This project is an **end-to-end Machine Learning system** designed to simulate and optimize Formula 1 race strategies.  
It leverages historical race data and tire degradation modeling to provide **data-driven decision support** for pit stop strategies and lap time prediction.

The system integrates:
- A trained **Random Forest regression model**
- A **real-time interactive dashboard (Streamlit)**
- A full **race strategy simulation engine**

---

## 🎯 Objectives

- Predict lap times based on race conditions  
- Simulate complete race strategies (multi-stint)  
- Optimize pit stop timing  
- Provide actionable insights to improve race performance  

---

## 🧠 Machine Learning Model

| Component | Description |
|----------|------------|
| Model | Random Forest Regressor |
| Task | Regression (Lap Time Prediction) |
| R² Score | ~0.93 |
| MAE | ~1 second |
| Dataset | FastF1 API |

### 📊 Feature Engineering

- Tire degradation modeling (linear + quadratic)
- Compound encoding (SOFT, MEDIUM, HARD)
- Fuel load approximation
- Driver encoding
- Interaction features (compound × tire age)
- Tire freshness indicator

---

## 🏗️ System Architecture
FastF1 API
↓
Data Preprocessing
↓
Feature Engineering
↓
ML Model (Random Forest)
↓
Strategy Simulation Engine
↓
Streamlit Dashboard UI

---

## 🚀 Key Features

### 🔮 Lap Time Prediction
Predict lap performance based on:
- Tire compound
- Tire age
- Lap number
- Driver
- Track conditions

---

### 🏁 Strategy Simulation
- Multi-stint race simulation  
- Custom strategies (1 to 3 pit stops)  
- Total race time computation  

---

### 🔍 Pit Stop Optimization
- Identify optimal pit window  
- Compare strategy outcomes  
- Quantify time gains/losses  

---

### 📈 Interactive Dashboard
- Real-time predictions  
- Strategy comparison visualizations  
- Performance metrics display  
- Interactive UI (Plotly + Streamlit)

---

## ⚙️ Tech Stack

| Category | Tools |
|--------|------|
| Language | Python |
| ML | Scikit-learn |
| Data | Pandas, NumPy |
| Visualization | Plotly |
| UI | Streamlit |
| Data Source | FastF1 API |
| Model Storage | Joblib |

---

## 📂 Project Structure
f1-strategy-optimizer/
│
├── app.py
├── models/
│ ├── lap_time_rf_model.pkl
│ └── model_metadata.pkl
├── data/
├── notebooks/
├── assets/
├── requirements.txt
└── README.md

---

## ▶️ Installation & Setup

```bash
git clone https://github.com/chahednefzi/f1-strategy-optimizer.git
cd f1-strategy-optimizer

python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows

pip install -r requirements.txt
▶️ Run the Application
streamlit run app.py

Then open:
👉 http://localhost:8501
Results & Visualizations

Add your screenshots here

Example:

![Dashboard](assets/dashboard.png)
![Strategy](assets/strategy.png)
![Optimizer](assets/optimizer.png)

📊 Example Use Cases

Compare 1-stop vs 2-stop strategies
Optimize pit stop timing
Analyze tire degradation impact
Evaluate driver performance scenarios

⚠️ Limitations
No weather or track evolution modeling
Limited SOFT tire data
Simplified fuel consumption model

🔮 Future Improvements
Weather data integration 🌧️
Deep Learning models (LSTM)
Real-time telemetry integration
Multi-driver race simulation
Reinforcement Learning for strategy optimization

👨‍💻 Author
Chahed Nefzi
Computer Science Student – AI & Data Engineering

⭐ Project Value

This project demonstrates:

End-to-end ML pipeline design
Feature engineering & modeling
Simulation system development
Data visualization & dashboard creation
Real-world problem solving