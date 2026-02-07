import streamlit as st
import numpy as np
import joblib

model = joblib.load("fraud_model.pkl")

st.title("?? Fraudulent Transaction Detection")

st.write("Enter transaction details")

features = []
for i in range(1, 29):
    features.append(st.number_input(f"V{i}", value=0.0))

time = st.number_input("Time", value=0.0)
amount = st.number_input("Amount", value=0.0)

input_data = np.array([features + [time, amount]])

if st.button("Predict"):
    result = model.predict(input_data)
    if result[0] == 1:
        st.error("?? Fraud Transaction")
    else:
        st.success("? Normal Transaction")
