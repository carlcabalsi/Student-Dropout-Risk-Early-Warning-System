import streamlit as st
import joblib

st.title("Student Dropout Risk Prediction")

model = joblib.load("models/dropout_model.pkl")

st.write("Model loaded successfully!")
