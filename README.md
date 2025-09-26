# 🌱 Crop Recommendation System

A machine learning-powered web application that recommends the most suitable crop based on soil nutrients, weather, and environmental conditions.

---

## 📖 Overview

This project uses multiple machine learning models (XGBoost, Random Forest, Decision Tree, Bagging Classifier) trained on historical crop data to predict the optimal crop. The app is built with **Streamlit** for an interactive web interface and allows users to input soil and weather parameters to get crop recommendations.

---

## 🔧 Features

- Predict the best crop based on:
  - Nitrogen (N)
  - Phosphorus (P)
  - Potassium (K)
  - Temperature (°C)
  - Humidity (%)
  - Soil pH
  - Rainfall (mm)
- Interactive Streamlit app for real-time recommendations
- Displays crop images for better visualization
- Model evaluation metrics included (accuracy of different models)

---

## 🛠️ Tech Stack

- **Python 3.x**
- **Libraries**: pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn, pillow, streamlit, requests
- **Machine Learning Models**: XGBoost, Random Forest, Decision Tree, Bagging Classifier
- **Deployment**: Streamlit / Render (can be deployed online)
- **Data**: `Crop_recommendation.csv`
- **Model Files**: `Tree.pkl` (trained ML models)

---


