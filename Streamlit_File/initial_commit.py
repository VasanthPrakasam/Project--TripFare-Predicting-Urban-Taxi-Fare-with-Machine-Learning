import streamlit as st
import pickle
import pandas as pd
import os

# Check if model files exist
if not os.path.exists('xgb_pipeline_model.pkl'):
    st.error("❌ Model file not found. Please ensure 'xgb_pipeline_model.pkl' is in the directory.")
    st.stop()

if not os.path.exists('label_encoders.pkl'):
    st.error("❌ Encoders file not found. Please ensure 'label_encoders.pkl' is in the directory.")
    st.stop()

# 🔁 Load the trained XGBoost pipeline model
with open('xgb_pipeline_model.pkl', 'rb') as file:
    model = pickle.load(file)

# 🔁 Load the label encoders
with open('label_encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

# 🎯 Extract specific encoders
pickup_area_encoder = encoders['pickup_area']
dropoff_area_encoder = encoders['dropoff_area']

# ✅ Feature order used during model training
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

# 🚖 App title
st.title("🚖 NYC Taxi Fare Predictor")

# Add model information
with st.expander("ℹ️ Model Information"):
    st.write("This XGBoost model predicts NYC taxi fares based on:")
    st.write("- Trip duration and distance")
    st.write("- Pickup and dropoff locations")
    st.write("- Time of day and passenger count")
    st.write("- Payment method")

# ⌨️ Collect user inputs
st.header("📥 Trip Details")

trip_duration = st.number_input("⏱️ Trip Duration (minutes)", min_value=1.0, value=10.0, step=1.0)
trip_distance = st.number_input("📏 Trip Distance (miles)", min_value=0.1, value=1.5, step=0.1)

# Fixed payment type selector
payment_options = {
    "Credit Card": 1,
    "Cash": 2, 
    "No Charge": 3,
    "Dispute": 4
}
payment_type_name = st.selectbox("💳 Payment Type", list(payment_options.keys()))
payment_type = payment_options[payment_type_name]

pickup_area_name = st.selectbox("📍 Pickup Area", pickup_area_encoder.classes_)
dropoff_area_name = st.selectbox("📍 Dropoff Area", dropoff_area_encoder.classes_)

pickup_hour = st.slider("🕐 Pickup Hour", 0, 23, 12)
dropoff_hour = st.slider("🕐 Dropoff Hour", 0, 23, 13)
passenger_count = st.number_input("👥 Passenger Count", min_value=1, max_value=8, value=1)

# 🧠 Encode categorical area names
pickup_area = pickup_area_encoder.transform([pickup_area_name])[0]
dropoff_area = dropoff_area_encoder.transform([dropoff_area_name])[0]

# Show input summary
st.header("📋 Trip Summary")
col1, col2 = st.columns(2)

with col1:
    st.write(f"**Duration:** {trip_duration} minutes")
    st.write(f"**Distance:** {trip_distance} miles")
    st.write(f"**Passengers:** {passenger_count}")
    st.write(f"**Payment:** {payment_type_name}")

with col2:
    st.write(f"**From:** {pickup_area_name}")
    st.write(f"**To:** {dropoff_area_name}")
    st.write(f"**Pickup Time:** {pickup_hour}:00")
    st.write(f"**Dropoff Time:** {dropoff_hour}:00")

# 📦 Form input DataFrame
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

# 🔮 Predict Fare
if st.button("Predict Fare 💰"):
    try:
        prediction = model.predict(input_data)[0]
        st.success(f"🚕 Estimated Fare: **${prediction:.2f}**")
        
        # Additional fare breakdown (optional)
        with st.expander("📊 Fare Breakdown"):
            base_rate = 2.50
            time_rate = prediction * 0.3
            distance_rate = prediction * 0.7
            
            st.write(f"**Estimated Components:**")
            st.write(f"- Base Rate: ${base_rate:.2f}")
            st.write(f"- Time Component: ${time_rate:.2f}")
            st.write(f"- Distance Component: ${distance_rate:.2f}")
            st.write(f"- **Total: ${prediction:.2f}**")
            
    except Exception as e:
        st.error(f"❌ Prediction Failed: {e}")
        st.write("Please check that all inputs are valid and the model files are properly loaded.")