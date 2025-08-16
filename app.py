import streamlit as st
import joblib
import numpy as np
from PIL import Image

# Load model and scaler
model = joblib.load("crop_model.joblib")
scaler = joblib.load("crop_scaler.joblib")

# Background image
bg_image = "farm_background.png"  # rename your selected image to this and upload

# Custom CSS
page_bg = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url("{bg_image}");
    background-size: cover;
    background-position: center;
}}

.container {{
    background-color: rgba(0, 128, 0, 0.8); /* green semi-transparent */
    padding: 20px;
    border-radius: 15px;
    width: 40%;
    margin-left: auto;  
    margin-right: 30px;
    color: black;
}}
h1, label {{
    color: black !important;
}}
.result-box {{
    background-color: white; /* solid white for clarity */
    border-radius: 12px;
    padding: 15px;
    text-align: center;
    margin-top: 20px;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.3);
}}
.result-text {{
    color: red;
    font-size: 22px;
    font-weight: bold;
}}
.result-icon {{
    font-size: 40px;
}}
</style>
"""

st.markdown(page_bg, unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align: center;'>🌱 Crop Recommendation System 🌱</h1>", unsafe_allow_html=True)

# Mapping crops to icons (add more as needed)
crop_icons = {
    "rice": "🌾",
    "maize": "🌽",
    "wheat": "🌿",
    "cotton": "🧵",
    "apple": "🍎",
    "banana": "🍌",
    "mango": "🥭",
    "grapes": "🍇",
    "coffee": "☕"
}

# Input form inside styled container
with st.container():
    st.markdown('<div class="container">', unsafe_allow_html=True)

    N = st.number_input("Nitrogen (N)", min_value=0, max_value=200, value=50)
    P = st.number_input("Phosphorus (P)", min_value=0, max_value=200, value=50)
    K = st.number_input("Potassium (K)", min_value=0, max_value=200, value=50)
    temperature = st.number_input("Temperature (°C)", min_value=0, max_value=50, value=25)
    humidity = st.number_input("Humidity (%)", min_value=0, max_value=100, value=50)
    ph = st.number_input("pH Value", min_value=0.0, max_value=14.0, value=6.5, step=0.1)
    rainfall = st.number_input("Rainfall (mm)", min_value=0, max_value=300, value=100)

    if st.button("🌾 Recommend Crop"):
        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]

        # Pick icon if available
        icon = crop_icons.get(prediction.lower(), "🌱")

        # White result box
        st.markdown(
            f"""
            <div class='result-box'>
                <div class='result-icon'>{icon}</div>
                <div class='result-text'>✅ Recommended Crop: {prediction}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('</div>', unsafe_allow_html=True)
