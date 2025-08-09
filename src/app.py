import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- Load model ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/best_rice_yield_model.joblib")
model = joblib.load(MODEL_PATH)

# --- Load historical data for base values ---
DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/processed/rice_data.csv")
df = pd.read_csv(DATA_PATH)
df = df[df['Year'] <= 2022]  # use only up to 2022

# --- Calculate base averages ---
BASE_TEMP = df['avg_temp'].mean()
BASE_RAIN = df['average_rain_fall_mm_per_year'].mean()
BASE_PEST = df['pesticides_tonnes'].mean()

# --- UI setup ---
st.set_page_config(page_title="Rice Yield Forecast", layout="centered")
st.title("🌾 Rice Yield Forecast (Nepal)")

st.markdown("""
Select a year between **2023 and 2035** to get the **predicted rice yield** based on climate trends:

- 📈 Temp: +0.2°C/year
- 💧 Rainfall: -1% per year
- 🧪 Pesticide use: +1.5% per year
""")

# --- Year selection ---
selected_year = st.slider("📅 Select Year", min_value=2023, max_value=2035, value=2025)

# --- Projected values ---
years_offset = selected_year - 2022
projected_temp = BASE_TEMP + years_offset * 0.2
projected_rain = BASE_RAIN * (0.99 ** years_offset)
projected_pest = BASE_PEST * (1.015 ** years_offset)

# --- Make prediction ---
input_df = pd.DataFrame([{
    'avg_temp': projected_temp,
    'average_rain_fall_mm_per_year': projected_rain,
    'pesticides_tonnes': projected_pest
}])

predicted_yield = model.predict(input_df)[0]

# --- Show result ---
st.subheader(f"📊 Predicted Yield for {selected_year}")
st.success(f"🌾 **{predicted_yield:.2f} hg/ha**")

with st.expander("📌 Show climate & pesticide inputs used"):
    st.write({
        "Projected Temperature (°C)": round(projected_temp, 2),
        "Projected Rainfall (mm/year)": round(projected_rain, 2),
        "Projected Pesticide Use (tonnes)": round(projected_pest, 2)
    })
