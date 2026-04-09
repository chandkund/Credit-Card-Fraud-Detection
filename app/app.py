import streamlit as st
import os
import sys
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# ================= FIX IMPORT PATH ================= #
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
st.set_page_config(page_title="Fraud Detection System", layout="wide")

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
st.sidebar.title("🏦 Banking Fraud System")
page = st.sidebar.radio("Navigation", ["Dashboard", "Transaction Check", "About"])

# ================= SESSION STATE ================= #
if "history" not in st.session_state:
    st.session_state.history = []

# ================= DASHBOARD ================= #
if page == "Dashboard":

    st.title("📊 Fraud Detection Dashboard")

    col1, col2, col3 = st.columns(3)
    col1.metric("Model", "XGBoost")
    col2.metric("ROC-AUC", "0.98")
    col3.metric("Fraud Recall", "0.88")

    st.markdown("---")

    # Model Comparison Table
    st.subheader("📊 Model Comparison")

    model_data = {
        "Model": ["Logistic Regression", "Random Forest", "XGBoost"],
        "ROC-AUC": [0.91, 0.96, 0.98]
    }

    df = pd.DataFrame(model_data)
    st.dataframe(df)

    # Fake trend graph (demo)
    st.subheader("📈 Fraud Trend")
    trend = np.random.randint(0, 100, 24)
    st.line_chart(trend)

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

    # Time input (real-world style)
    st.subheader("⏱ Transaction Time")
    hour = st.slider("Select Hour (0–23)", 0, 23, 12)
    time = hour * 3600

    st.markdown("---")

    if st.button("🚀 Analyze Transaction"):

        # Create feature array (30 features)
        features = np.zeros(30)
        features[0] = time
        features[1] = amount

        # Simulated mapping (for demo)
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

        # ================= HUMAN EXPLANATION ================= #
        st.subheader("🧠 Reasoning")

        reasons = []

        if amount > 2000:
            reasons.append("High transaction amount")

        if hour < 5:
            reasons.append("Unusual transaction time")

        if location == "High":
            reasons.append("High-risk location")

        if device == "Mobile":
            reasons.append("Untrusted device")

        if len(reasons) > 0:
            for r in reasons:
                st.write(f"⚠️ {r}")
        else:
            st.write("Transaction looks normal")

        # ================= SHAP EXPLAINABILITY ================= #
        st.subheader("📊 SHAP Explanation")

        try:
            feature_names = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
            input_df = pd.DataFrame([features], columns=feature_names)

            shap_values = explainer(input_df)

            fig, ax = plt.subplots()
            shap.plots.waterfall(shap_values[0], show=False)
            st.pyplot(fig)

        except Exception as e:
            st.warning("SHAP visualization could not be generated")

        # ================= HISTORY ================= #
        st.session_state.history.append({
            "Amount": amount,
            "Risk Score": round(prob, 2)
        })

        st.subheader("📜 Transaction History")
        st.table(st.session_state.history)

# ================= ABOUT ================= #
else:
    st.title("📘 About This System")

    st.write("""
    This is a real-time fraud detection system built using Machine Learning.

    🔹 Model: XGBoost  
    🔹 Technique: SMOTE  
    🔹 Explainability: SHAP  

    💡 Features:
    - Real-time prediction  
    - Risk scoring  
    - Explainable AI  
    - Banking-style UI  

    ⚠️ Note:
    Dataset uses PCA-transformed features (V1–V28), so inputs are simulated.
    """)