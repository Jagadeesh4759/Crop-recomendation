import streamlit as st
import joblib
import numpy as np
import base64

# --------------------------
# Set Page Config
# --------------------------
st.set_page_config(page_title="Crop Prediction App", layout="wide")

# --------------------------
# Function to set background
# --------------------------
def add_bg_from_local(image_file):
    with open(image_file, "rb") as img:
        encoded_string = base64.b64encode(img.read())
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{encoded_string.decode()});
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Call background function
add_bg_from_local("farm_background.png")

# --------------------------
# Load model & scaler
# --------------------------
model = joblib.load("crop_model.joblib")
scaler = joblib.load("crop_scaler.joblib")

# Crop labels + icons
crop_dict = {
    "rice": "🌾",
    "maize": "🌽",
    "chickpea": "🍲",
    "kidneybeans": "🫘",
    "pigeonpeas": "🥘",
    "mothbeans": "🌱",
    "mungbean": "🍵",
    "blackgram": "⚫",
    "lentil": "🥣",
    "pomegranate": "🍎",
    "banana": "🍌",
    "mango": "🥭",
    "grapes": "🍇",
    "watermelon": "🍉",
    "muskmelon": "🍈",
    "apple": "🍏",
    "orange": "🍊",
    "papaya": "🥥",
    "coconut": "🥥",
    "cotton": "👕",
    "jute": "🧵",
    "coffee": "☕"
}

# --------------------------
# UI Design
# --------------------------
st.markdown(
    """
    <style>
    .main-container {
        background-color: rgba(0, 128, 0, 0.75);  /* green with transparency */
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.4);
        color: black;
        max-width: 500px;
        margin: auto;   /* center the card */
    }
    .title {
        color: white;
        text-align: center;
        font-size: 28px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .result-box {
        background-color: white;
        color: black;
        padding: 15px;
        border-radius: 15px;
        text-align: center;
        font-size: 20px;
        font-weight: bold;
        margin-top: 20px;
        box-shadow: 0px 2px 6px rgba(0,0,0,0.3);
    }
    </style>
    """,
    unsafe_allow_html=True
)

with st.container():
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.markdown('<p class="title">🌱 Smart Crop Prediction 🌱</p>', unsafe_allow_html=True)

    # Input fields
    N = st.number_input("Nitrogen (N)", min_value=0, max_value=200, value=50)
    P = st.number_input("Phosphorus (P)", min_value=0, max_value=200, value=50)
    K = st.number_input("Potassium (K)", min_value=0, max_value=200, value=50)
    temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)
    ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=300.0, value=100.0)

    if st.button("Predict Crop"):
        # Preprocess input
        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        scaled = scaler.transform(features)
        prediction = model.predict(scaled)[0]

        # Show result inside styled box
        st.markdown(
            f'<div class="result-box">🌿 Recommended Crop: <br><br> {prediction.capitalize()} {crop_dict.get(prediction, "")}</div>',
            unsafe_allow_html=True
        )

    st.markdown('</div>', unsafe_allow_html=True)
