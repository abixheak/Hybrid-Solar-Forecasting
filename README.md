# ☀️ Advanced Hybrid Solar Forecasting

An industry-grade machine learning pipeline that predicts solar irradiance (W/m²) by combining the linear baseline stability of **SARIMAX** with the non-linear anomaly-correction of a Deep Learning **LSTM**.

## 📖 Project Overview
Accurate solar forecasting is critical for renewable energy grid stability. This project tackles the limitations of standard statistical models (which fail during sudden weather shifts) by deploying a dual-engine architecture:
1. **SARIMAX (Statistical):** Maps the predictable, daily, and seasonal baseline solar cycle.
2. **LSTM (Deep Learning):** Analyzes the most recent 24 hours of weather context to predict and correct the exact residual errors the SARIMAX model is about to make.

## ✨ Key Features
* **Hybrid Architecture:** Outperforms standalone statistical models with a **3.7% reduction in major errors (RMSE)** during sudden weather shifts.
* **Live API Integration:** Fetches real-time, 24-hour future weather data via the Open-Meteo API for dynamic live forecasting.
* **Smart Feature Engineering:** Utilizes cyclical time encodings (Sine/Cosine mapping) and custom "Cloud Effect" proxies.
* **Interactive Dashboard:** Deployed via Streamlit, allowing users to toggle between live API feeds and manual weather override simulations.

## 🏗️ Project Architecture
```text
Hybrid_Solar_Project/
├── data/
│   ├── raw/                  # Raw NASA POWER satellite data
│   ├── processed/            # Cleaned data & engineered features
├── models/
│   ├── saved_models/         # Pre-trained .pkl and .keras artifacts
├── assets/                   # Evaluation charts and diagrams
├── app.py                    # Streamlit deployment application
├── requirements.txt
└── README.md
```
