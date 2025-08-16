import streamlit as st
import joblib
import numpy as np

# Load the model and scaler
model = joblib.load("crop_model.joblib")
scaler = joblib.load("crop_scaler.joblib")

# Page config
st.set_page_config(page_title="🌱 Crop Recommendation System", page_icon="🌾", layout="centered")

# Custom CSS
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f8ff;
        background-image: linear-gradient(to bottom right, #f0f8ff, #e6f7ff);
    }
    .stTextInput>div>div>input, .stNumberInput>div>div>input {
        border: 2px solid #006400;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #228B22;
        color: white;
        border-radius: 10px;
        padding: 0.5em 1em;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #006400;
        color: #fff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Crop images dictionary (replace URLs with your own images if needed)
crop_images = {
    "rice": ("🌾", "https://upload.wikimedia.org/wikipedia/commons/6/6f/Rice_plants_%28IRRI%29.jpg"),
    "maize": ("🌽", "https://upload.wikimedia.org/wikipedia/commons/7/7b/Corncobs.jpg"),
    "chickpea": ("🫘", "https://upload.wikimedia.org/wikipedia/commons/2/27/Chickpeas.jpg"),
    "kidneybeans": ("🫘", "https://upload.wikimedia.org/wikipedia/commons/2/26/Kidney_beans.jpg"),
    "pigeonpeas": ("🫘", "https://upload.wikimedia.org/wikipedia/commons/6/6a/Pigeon_peas.jpg"),
    "mothbeans": ("🫘", "https://upload.wikimedia.org/wikipedia/commons/4/47/Moth_beans.jpg"),
    "mungbean": ("🫘", "https://upload.wikimedia.org/wikipedia/commons/f/f2/Mung_bean.jpg"),
    "blackgram": ("🫘", "https://upload.wikimedia.org/wikipedia/commons/5/5d/Urad_Dal.jpg"),
    "lentil": ("🥙", "https://upload.wikimedia.org/wikipedia/commons/4/45/Red_lentils.jpg"),
    "pomegranate": ("🍎", "https://upload.wikimedia.org/wikipedia/commons/9/91/Pomegranate_fruit_-_whole_and_piece_with_arils.jpg"),
    "banana": ("🍌", "https://upload.wikimedia.org/wikipedia/commons/8/8a/Banana-Single.jpg"),
    "mango": ("🥭", "https://upload.wikimedia.org/wikipedia/commons/9/90/Hapus_Mango.jpg"),
    "grapes": ("🍇", "https://upload.wikimedia.org/wikipedia/commons/1/1a/Table_grapes_on_white.jpg"),
    "watermelon": ("🍉", "https://upload.wikimedia.org/wikipedia/commons/f/f2/Watermelon.jpg"),
    "muskmelon": ("🍈", "https://upload.wikimedia.org/wikipedia/commons/1/16/Muskmelon.jpg"),
    "apple": ("🍎", "https://upload.wikimedia.org/wikipedia/commons/1/15/Red_Apple.jpg"),
    "orange": ("🍊", "https://upload.wikimedia.org/wikipedia/commons/c/c4/Orange-Fruit-Pieces.jpg"),
    "papaya": ("🍍", "https://upload.wikimedia.org/wikipedia/commons/7/7e/Papaya_cross_section_BNC.jpg"),
    "coconut": ("🥥", "https://upload.wikimedia.org/wikipedia/commons/1/15/Coconut_on_white_background.jpg"),
    "cotton": ("🧵", "https://upload.wikimedia.org/wikipedia/commons/f/f8/CottonPlant.JPG"),
    "jute": ("🧶", "https://upload.wikimedia.org/wikipedia/commons/6/65/Jute_Plant.jpg"),
    "coffee": ("☕", "https://upload.wikimedia.org/wikipedia/commons/4/45/Coffea_arabica_-_K%C3%B6hler%E2%80%93s_Medizinal-Pflanzen-037.jpg")
}

# Title
st.markdown("<h1 style='text-align: center; color: #006400;'>🌱 Crop Recommendation System</h1>", unsafe_allow_html=True)
st.write("### Fill in the soil and climate details to get the best crop recommendation.")

# Input form
with st.form("crop_form"):
    N = st.number_input("Nitrogen", min_value=0, max_value=140, value=50)
    P = st.number_input("Phosphorus", min_value=5, max_value=145, value=40)
    K = st.number_input("Potassium", min_value=5, max_value=205, value=35)
    temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=80.0)
    ph = st.number_input("pH Value", min_value=0.0, max_value=14.0, value=6.5)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=300.0, value=100.0)

    submit = st.form_submit_button("🌾 Predict Crop")

# Prediction
if submit:
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)[0]

    # Get emoji & image if available
    emoji, img_url = crop_images.get(prediction, ("🌿", ""))

    st.success(f"✅ Recommended Crop: **{emoji} {prediction.capitalize()}**")
    if img_url:
        st.image(img_url, caption=f"Recommended Crop: {prediction.capitalize()}", use_column_width=True)

    st.balloons()
