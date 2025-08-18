import streamlit as st
import pickle
import pandas as pd

# 🔁 Load the trained pipeline model
with open('xgb_pipeline_model.pkl', 'rb') as file:
    model = pickle.load(file)

# ✅ Define the correct feature names (order matters!)
feature_names = [
    'trip_duration_minutes',
    'trip_distance_miles',
    'payment_type',
    'pickup_area',
    'dropoff_area',
    'pickup_hour',
    'dropoff_hour',
    'passenger_count'
]

st.title("🚖 NYC Taxi Fare Predictor")

# ⌨️ User Inputs (match the features)
trip_duration = st.number_input("⏱️ Trip Duration (minutes)", value=10.0)
trip_distance = st.number_input("📏 Trip Distance (miles)", value=1.5)
payment_type = st.selectbox("💳 Payment Type (Encoded)", [0, 1, 2, 3, 4, 5])
pickup_area = st.selectbox("📍 Pickup Area (Encoded)", list(range(0, 265)))
dropoff_area = st.selectbox("📍 Dropoff Area (Encoded)", list(range(0, 265)))
pickup_hour = st.slider("🕐 Pickup Hour", 0, 23, 12)
dropoff_hour = st.slider("🕐 Dropoff Hour", 0, 23, 13)
passenger_count = st.number_input("👥 Passenger Count", min_value=1, max_value=8, value=1)

# 📦 Collect into a DataFrame in the correct order
input_data = pd.DataFrame([[
    trip_duration,
    trip_distance,
    payment_type,
    pickup_area,
    dropoff_area,
    pickup_hour,
    dropoff_hour,
    passenger_count
]], columns=feature_names)

# 🔮 Prediction
if st.button("Predict Fare 💰"):
    fare_prediction = model.predict(input_data)[0]
    st.success(f"🚕 Estimated Fare: **${fare_prediction:.2f}**")


