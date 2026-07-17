import streamlit as st
import pandas as pd
import joblib

# ==========================
# Load Model
# ==========================
model = joblib.load("svm_fuel_model.pkl")
scaler = joblib.load("scaler.pkl")
ticker_encoder = joblib.load("ticker_encoder.pkl")
commodity_encoder = joblib.load("commodity_encoder.pkl")

# ==========================
# Page Configuration
# ==========================
st.set_page_config(
    page_title="Fuel Price Prediction",
    page_icon="⛽",
    layout="wide"
)

# ==========================
# Custom CSS
# ==========================
st.markdown("""
<style>
.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
}

.stButton > button{
    width:100%;
    border-radius:12px;
    height:55px;
    font-size:18px;
    font-weight:bold;
}

div[data-testid="stMetric"]{
    border-radius:15px;
    padding:20px;
    box-shadow:0 2px 10px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# ==========================
# Header
# ==========================
st.title("⛽ Fuel Price Prediction System")

st.markdown("""
### ระบบทำนายแนวโน้มราคาพลังงานด้วย Machine Learning

ระบบนี้ใช้โมเดล **Support Vector Machine (SVM)** สำหรับวิเคราะห์ข้อมูลตลาดและทำนายว่า

- 📈 ราคาปิดวันถัดไปมีแนวโน้มเพิ่มขึ้น
- 📉 ราคาปิดวันถัดไปมีแนวโน้มลดลง

กรอกข้อมูลด้านล่างและกดปุ่ม **Predict Fuel Trend**
""")

st.divider()

# ==========================
# Sidebar
# ==========================
st.sidebar.title("ℹ️ About Project")

st.sidebar.info("""
**Machine Learning Model**
- Support Vector Machine (SVM)

**Deployment**
- Streamlit

**Dataset**
- Fuel Market Dataset

**Features**
- Open Price
- High Price
- Low Price
- Close Price
- Volume
""")

# ==========================
# Input Section
# ==========================
col1, col2 = st.columns(2)

with col1:
    st.subheader("📌 Market Information")

    ticker = st.selectbox(
        "Ticker",
        ticker_encoder.classes_
    )

    commodity = st.selectbox(
        "Commodity",
        commodity_encoder.classes_
    )

    volume = st.number_input(
        "Trading Volume",
        min_value=0.0,
        value=0.0,
        format="%.2f"
    )

with col2:
    st.subheader("💰 Price Information")

    open_price = st.number_input(
        "Open Price",
        min_value=0.0,
        value=0.0,
        format="%.2f"
    )

    high_price = st.number_input(
        "High Price",
        min_value=0.0,
        value=0.0,
        format="%.2f"
    )

    low_price = st.number_input(
        "Low Price",
        min_value=0.0,
        value=0.0,
        format="%.2f"
    )

    close_price = st.number_input(
        "Close Price",
        min_value=0.0,
        value=0.0,
        format="%.2f"
    )

st.divider()

# ==========================
# Date Input
# ==========================
st.subheader("📅 Date Information")

col3, col4, col5 = st.columns(3)

with col3:
    year = st.number_input(
        "Year",
        min_value=2000,
        max_value=2100,
        value=2026
    )

with col4:
    month = st.number_input(
        "Month",
        min_value=1,
        max_value=12,
        value=7
    )

with col5:
    day = st.number_input(
        "Day",
        min_value=1,
        max_value=31,
        value=17
    )

st.write("")

# ==========================
# Prediction
# ==========================
if st.button("🚀 Predict Fuel Trend"):

    input_df = pd.DataFrame([[
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
        "ticker",
        "commodity",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "year",
        "month",
        "day"
    ])

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)

    st.divider()

    st.subheader("📊 Prediction Result")

    if prediction[0] == 1:

        st.success(
            "📈 Prediction Result : ราคามีแนวโน้มเพิ่มขึ้นในวันถัดไป"
        )

        st.metric(
            label="Predicted Trend",
            value="UP"
        )

        st.markdown("""
### รายละเอียดผลการทำนาย

โมเดล SVM วิเคราะห์รูปแบบของราคาและปริมาณการซื้อขายแล้วพบว่า

✅ ตลาดมีแนวโน้มเป็นขาขึ้น

ข้อมูลในอดีตที่ใกล้เคียงกับข้อมูลชุดนี้มักนำไปสู่การเพิ่มขึ้นของราคาในวันถัดไป
""")

    else:

        st.error(
            "📉 Prediction Result : ราคามีแนวโน้มลดลงในวันถัดไป"
        )

        st.metric(
            label="Predicted Trend",
            value="DOWN"
        )

        st.markdown("""
### รายละเอียดผลการทำนาย

โมเดล SVM วิเคราะห์รูปแบบของราคาและปริมาณการซื้อขายแล้วพบว่า

⚠️ ตลาดมีแนวโน้มเป็นขาลง

ข้อมูลในอดีตที่ใกล้เคียงกับข้อมูลชุดนี้มักนำไปสู่การลดลงของราคาในวันถัดไป
""")

    st.info("""
ผลลัพธ์นี้เป็นการคาดการณ์จาก Machine Learning เพื่อการศึกษาและการทดลองเท่านั้น
ไม่ใช่คำแนะนำด้านการลงทุนหรือการซื้อขายจริง
""")

# ==========================
# Footer
# ==========================
st.divider()

st.caption(
    "Developed with Streamlit and Scikit-learn | Support Vector Machine (SVM)"
)

