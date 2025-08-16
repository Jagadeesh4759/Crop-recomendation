import streamlit as st
import joblib, numpy as np

model = joblib.load("crop_model.joblib")
scaler = joblib.load("crop_scaler.joblib")
FEATURES = ['N','P','K','temperature','humidity','ph','rainfall']

st.set_page_config(page_title="Crop Recommendation", page_icon="🌾")
st.title("🌱 Crop Recommendation System (Colab)")

N = st.number_input("Nitrogen (N)", 0, 300, 90)
P = st.number_input("Phosphorus (P)", 0, 300, 42)
K = st.number_input("Potassium (K)", 0, 300, 43)
temperature = st.number_input("Temperature (°C)", 0.0, 60.0, 20.0)
humidity = st.number_input("Humidity (%)", 0.0, 100.0, 82.0)
ph = st.number_input("pH", 0.0, 14.0, 6.5)
rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, 202.0)

def predict_top3(values):
    x = np.array([values], dtype=float)
    x_s = scaler.transform(x)
    pred = model.predict(x_s)[0]
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x_s)[0]
        classes = model.classes_
        idx = np.argsort(proba)[::-1][:3]
        top3 = [(classes[i], float(proba[i])) for i in idx]
    else:
        top3 = [(pred, 1.0)]
    return pred, top3

if st.button("Recommend Crop"):
    vals = [N, P, K, temperature, humidity, ph, rainfall]
    try:
        pred, top3 = predict_top3(vals)
        st.success(f"🌾 Recommended Crop: **{pred}**")
        st.subheader("Top-3 (with probabilities)")
        for name, p in top3:
            st.write(f"- {name}: {p:.3f}")
        with st.expander("Feature order used"):
            st.write(FEATURES)
    except Exception as e:
        st.error(f"Error: {e}")
