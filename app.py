import streamlit as st
import pandas as pd
import joblib

# =====================
# Load Model
# =====================
model = joblib.load("svm_fuel_model.pkl")
scaler = joblib.load("scaler.pkl")
ticker_encoder = joblib.load("ticker_encoder.pkl")
commodity_encoder = joblib.load("commodity_encoder.pkl")

# =====================
# Streamlit UI
# =====================
st.set_page_config(
    page_title="Fuel Price Prediction",
    page_icon="⛽",
    layout="centered"
)

st.title("⛽ Fuel Price Prediction using SVM")
st.write("ทำนายว่าราคาปิดของวันถัดไปจะเพิ่มขึ้นหรือลดลง")

# =====================
# Input
# =====================
ticker = st.selectbox(
    "Ticker",
    ticker_encoder.classes_
)

commodity = st.selectbox(
    "Commodity",
    commodity_encoder.classes_
)

open_price = st.number_input(
    "Open Price",
    min_value=0.0,
    format="%.2f"
)

high_price = st.number_input(
    "High Price",
    min_value=0.0,
    format="%.2f"
)

low_price = st.number_input(
    "Low Price",
    min_value=0.0,
    format="%.2f"
)

close_price = st.number_input(
    "Close Price",
    min_value=0.0,
    format="%.2f"
)

volume = st.number_input(
    "Volume",
    min_value=0.0,
    format="%.2f"
)

year = st.number_input(
    "Year",
    min_value=2000,
    max_value=2100,
    value=2026
)

month = st.number_input(
    "Month",
    min_value=1,
    max_value=12,
    value=7
)

day = st.number_input(
    "Day",
    min_value=1,
    max_value=31,
    value=17
)

# =====================
# Prediction
# =====================
if st.button("Predict"):

    input_data = pd.DataFrame([[
        ticker_encoder.transform([ticker])[0],
        commodity_encoder.transform([commodity])[0],
        open_price,
        high_price,
        low_price,
        close_price,
        volume,
        year,
        month,
        day
    ]], columns=[
        'ticker',
        'commodity',
        'open',
        'high',
        'low',
        'close',
        'volume',
        'year',
        'month',
        'day'
    ])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.success("📈 Prediction: ราคามีแนวโน้มเพิ่มขึ้น")
    else:
        st.error("📉 Prediction: ราคามีแนวโน้มลดลง")