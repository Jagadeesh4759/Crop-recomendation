import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("crop_model.joblib")
scaler = joblib.load("crop_scaler.joblib")

# Custom CSS for styling
st.markdown(
    """
    <style>
    .stApp {
        background: url("farm_background.png");
        background-size: cover;
        background-position: center;
    }
    .main-container {
        background-color: rgba(0, 128, 0, 0.75);  /* semi-transparent green */
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.4);
        color: black;
        max-width: 500px;
        margin: auto;
    }
    .header-bar {
        background-color: #006400;   /* dark green */
        padding: 15px;
        border-top-left-radius: 20px;
        border-top-right-radius: 20px;
        text-align: center;
    }
    .header-bar h1 {
        color: white;
        font-size: 26px;
        margin: 0;
    }
    .content {
        padding-top: 20px;
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

    /* 🔴 Make all entered input values red, bold, bigger */
    div[data-baseweb="input"] input[type="number"],
    div[data-baseweb="input"] input[type="text"] {
        color: red !important;
        font-size: 16px !important;
        font-weight: bold !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Crop emojis dictionary
crop_emojis = {
    "rice": "🌾",
    "maize": "🌽",
    "chickpea": "🍲",
    "kidneybeans": "🫘",
    "pigeonpeas": "🌿",
    "mothbeans": "🫘",
    "mungbean": "🫘",
    "blackgram": "🫘",
    "lentil": "🥣",
    "pomegranate": "🍎",
    "banana": "🍌",
    "mango": "🥭",
    "grapes": "🍇",
    "watermelon": "🍉",
    "muskmelon": "🍈",
    "apple": "🍏",
    "orange": "🍊",
    "papaya": "🥭",
    "coconut": "🥥",
    "cotton": "👕",
    "jute": "🧵",
    "coffee": "☕"
}

# App Layout
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Header bar
st.markdown('<div class="header-bar"><h1>🌱 Smart Crop Prediction 🌱</h1></div>', unsafe_allow_html=True)

st.markdown('<div class="content">', unsafe_allow_html=True)

# Input fields
N = st.number_input("Nitrogen (N)", min_value=0.0, step=1.0)
P = st.number_input("Phosphorus (P)", min_value=0.0, step=1.0)
K = st.number_input("Potassium (K)", min_value=0.0, step=1.0)
temperature = st.number_input("Temperature (°C)", step=0.1)
humidity = st.number_input("Humidity (%)", step=0.1)
ph = st.number_input("Soil pH", step=0.1)
rainfall = st.number_input("Rainfall (mm)", step=0.1)

if st.button("Predict Crop"):
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)[0]

    emoji = crop_emojis.get(prediction, "🌱")
    st.markdown(
        f'<div class="result-box">Recommended Crop: {prediction.capitalize()} {emoji}</div>',
        unsafe_allow_html=True
    )

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
