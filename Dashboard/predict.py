import streamlit as st
import joblib
import numpy as np

# Load your trained model, encoder, and scaler
model = joblib.load("models/rf_model.pkl")           # path to your Random Forest model
encoder = joblib.load("models/encoder.pkl")          # OrdinalEncoder
scaler = joblib.load("models/scaler.pkl")            # StandardScaler

# Label map to convert numeric predictions back to platform names
platform_labels = {0: "Facebook", 1: "Instagram", 2: "LinkedIn", 3: "Snapchat", 4: "Twitter"}

def show_predictor_page():
    st.title("📊 Predict Social Media Platform")
    st.write("Enter user details to predict the likely social media platform.")

    # User Inputs
    age = st.number_input("Age", min_value=10, max_value=100, value=25)
    gender = st.selectbox("Gender", ["male", "female", "non-binary"])
    daily_usage = st.number_input("Daily Usage Time (minutes)", value=120)
    posts = st.number_input("Posts Per Day", value=2)
    likes = st.number_input("Likes Received Per Day", value=50)
    comments = st.number_input("Comments Received Per Day", value=5)
    messages = st.number_input("Messages Sent Per Day", value=10)

    if st.button("Predict"):
        # Encode categorical
        gender_encoded = encoder.transform([[gender, "Instagram", "Happy", "18-24"]])[0][0]  # dummy values for unused columns

        # Form feature array: match training order
        input_array = np.array([[age, gender_encoded, daily_usage, posts, likes, comments, messages]])

        # Scale
        input_scaled = scaler.transform(input_array)

        # Predict
        prediction = model.predict(input_scaled)[0]
        platform = platform_labels.get(prediction, "Unknown")

        st.success(f"🎯 Predicted Platform: **{platform}**")

    if st.button("🔙 Back to Home"):
        st.query_params.page = "home"
        st.rerun()
