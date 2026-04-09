import streamlit as st
import os
import sys
import numpy as np
import shap
import matplotlib.pyplot as plt

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.predict import load_artifacts, predict

# ================= PATH SETUP ================= #
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

# ================= LOAD MODEL ================= #
model, scaler = load_artifacts(MODEL_PATH, SCALER_PATH)

# SHAP Explainer
explainer = shap.Explainer(model)

# ================= PAGE CONFIG ================= #
st.set_page_config(page_title="Bank Fraud Detection", layout="wide")

# ================= UI STYLE ================= #
st.markdown("""
<style>
.stApp {
    background-color: #0e1117;
}
h1, h2, h3 {
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ================= SIDEBAR ================= #
st.sidebar.title("🏦 Banking System")
page = st.sidebar.radio("Navigation", ["Dashboard", "Transaction Check", "About"])

# ================= DASHBOARD ================= #
if page == "Dashboard":
    st.title("📊 Fraud Detection Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric("Model", "XGBoost")
    col2.metric("ROC-AUC", "0.98")
    col3.metric("Fraud Recall", "0.88")

    st.markdown("---")

    st.subheader("📈 Fraud Trend (Demo)")
    data = np.random.randint(0, 100, 24)
    st.line_chart(data)

# ================= TRANSACTION PAGE ================= #
elif page == "Transaction Check":

    st.title("💳 Transaction Monitoring System")

    st.subheader("💳 Transaction Details")

    col1, col2 = st.columns(2)

    with col1:
        amount = st.number_input("💰 Amount (₹)", min_value=0.0, value=500.0)
        transaction_type = st.selectbox("Transaction Type", ["Online", "POS", "ATM"])

    with col2:
        location = st.selectbox("Location Risk", ["Low", "Medium", "High"])
        device = st.selectbox("Device", ["Mobile", "Laptop", "ATM"])

    st.subheader("⏱ Transaction Time")

    hour = st.slider("Select Hour (0–23)", 0, 23, 12)
    time = hour * 3600

    st.markdown("---")

    if st.button("🚀 Analyze Transaction"):

        # Create feature array
        features = np.zeros(30)
        features[0] = time
        features[1] = amount

        # Simulated feature mapping
        if transaction_type == "Online":
            features[2] = 1.5
        if location == "High":
            features[3] = 2.0
        if device == "Mobile":
            features[4] = 1.2

        # Prediction
        pred, prob = predict(model, scaler, features)

        st.markdown("## 🔍 Prediction Result")

        if pred == 1:
            st.error(f"🚨 Fraud Detected (Risk: {prob:.2f})")
        else:
            st.success(f"✅ Legitimate Transaction (Risk: {prob:.2f})")

        # ================= RISK METER ================= #
        st.subheader("⚠️ Risk Score")
        st.progress(int(prob * 100))

        if prob > 0.7:
            st.warning("High Risk Transaction")
        elif prob > 0.4:
            st.info("Medium Risk")
        else:
            st.success("Low Risk")

        # ================= SHAP EXPLAINABILITY ================= #
        st.subheader("🧠 Why this prediction?")

        try:
            shap_values = explainer(features.reshape(1, -1))

            fig, ax = plt.subplots()
            shap.plots.waterfall(shap_values[0], show=False)
            st.pyplot(fig)

        except Exception as e:
            st.error(f"SHAP error: {e}")

# ================= ABOUT ================= #
else:
    st.title("📘 About This System")

    st.write("""
    This is a real-time fraud detection system built using Machine Learning.

    🔹 Model: XGBoost  
    🔹 Technique: SMOTE (handles imbalance)  
    🔹 Explainability: SHAP  

    💡 Features:
    - Real-time prediction  
    - Risk scoring system  
    - Explainable AI  
    - Banking-style UI  

    ⚠️ Note:
    Dataset uses PCA-transformed features (V1–V28), so inputs are simulated for demo.
    """)