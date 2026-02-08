import streamlit as st
import numpy as np
import joblib

# Page config
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="💳",
    layout="wide"
)

# Load model
model = joblib.load("fraud_model.pkl")

# ---------- HEADER ----------
st.markdown(
    "<h1 style='text-align: center;'>💳 Fraudulent Transaction Detection System</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center;'>Machine Learning based real-time fraud prediction</p>",
    unsafe_allow_html=True
)

st.write("---")

# ---------- SIDEBAR ----------
st.sidebar.header("📌 Project Info")
st.sidebar.write("""
**Domain:** Machine Learning  
**Model:** XGBoost  
**Dataset:** Credit Card Fraud (Kaggle)  
**Output:** Fraud / Legitimate
""")

st.sidebar.header("⚙️ Instructions")
st.sidebar.write("""
- Enter transaction values  
- Click **Predict Transaction**  
- View result instantly  
""")

# ---------- MAIN LAYOUT ----------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🧾 Enter Transaction Details")

    features = []
    for i in range(1, 29):
        value = st.number_input(f"V{i}", value=0.0, format="%.4f")
        features.append(value)

    time = st.number_input("Time", value=0.0)
    amount = st.number_input("Amount", value=0.0)

    input_data = np.array([features + [time, amount]])

    predict_btn = st.button("🔍 Predict Transaction")

with col2:
    st.subheader("📊 Prediction Result")
    st.info("Result will be shown here after prediction")

    if predict_btn:
        prediction = model.predict(input_data)

        if prediction[0] == 1:
            st.error("🚨 **Fraudulent Transaction Detected**")
            st.markdown("""
            **Action Suggested:**  
            - Block transaction  
            - Verify user identity  
            """)
        else:
            st.success("✅ **Legitimate Transaction**")
            st.markdown("""
            **Status:**  
            - Transaction is safe  
            """)

st.write("---")

# ---------- FOOTER ----------
st.markdown(
    "<p style='text-align: center; font-size: 13px;'>MCA Minor Project | Fraud Detection using Machine Learning</p>",
    unsafe_allow_html=True
)


