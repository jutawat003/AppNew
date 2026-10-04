import glob
import os
import numpy as np
import pandas as pd
import pickle
import streamlit as st
import Orange

# ตั้งค่าหน้าเว็บ Streamlit
st.set_page_config(
    page_title="AI แนะนำอาชีพเสริมทางการเงิน", page_icon="💼", layout="centered"
)

st.title("💼 AI แนะนำอาชีพเสริมที่เหมาะสมตามศักยภาพทางการเงิน")
st.write(
    "ระบบประเมินจากโมเดล Linear Regression ที่เทรนจาก Orange Data Mining"
)

# หาตำแหน่ง Directory ปัจจุบัน
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ฟังก์ชั่นค้นหาไฟล์โมเดลอัตโนมัติ
def find_model_file():
    candidates = (
        glob.glob(os.path.join(BASE_DIR, "*.pkds"))
        + glob.glob(os.path.join(BASE_DIR, "*.pkcls"))
        + glob.glob(os.path.join(BASE_DIR, "*.pkl"))
        + glob.glob(os.path.join(BASE_DIR, "*.sav"))
    )
    if candidates:
        return candidates[0]
    return None


# โหลดโมเดล Orange
@st.cache_resource
def load_model():
    model_path = find_model_file()
    if not model_path:
        st.error("❌ หาไฟล์โมเดลไม่พบ!")
        st.stop()

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


model = load_model()

# ส่วนรับข้อมูลจากผู้ใช้
st.header("1. กรอกข้อมูลทางการเงินของคุณ")
col1, col2 = st.columns(2)

with col1:
    monthly_income = st.number_input(
        "รายได้ต่อเดือน (Monthly Income)", value=80000.0, step=1000.0
    )
    monthly_expenditure = st.number_input(
        "ค่าใช้จ่ายต่อเดือน (Monthly Expenditure)", value=40000.0, step=1000.0
    )
    market_volatility = st.slider(
        "ระดับความผันผวนของตลาดที่ยอมรับได้ (Market Volatility)",
        10.0,
        80.0,
        30.0,
    )
    inflation_rate = st.slider(
        "อัตราเงินเฟ้อปัจจุบัน (%)", 2.0, 8.0, 4.0, step=0.1
    )
    investment_amount = st.number_input(
        "เงินทุนที่พร้อมลงทุน (Investment Amount)", value=50000.0, step=1000.0
    )
    savings_ratio = st.slider(
        "สัดส่วนเงินออม (Savings Ratio)", 0.0, 0.6, 0.3, step=0.01
    )

with col2:
    credit_score = st.slider("คะแนนเครดิต (Credit Score)", 550, 850, 700)
    debt_to_income = st.slider(
        "อัตราส่วนหนี้สินต่อรายได้ (DTI)", 0.1, 0.7, 0.3, step=0.01
    )
    risk_tolerance = st.slider(
        "ระดับการยอมรับความเสี่ยง (Risk Tolerance)", 0.0, 1.0, 0.5, step=0.01
    )
    economic_sentiment = st.slider(
        "มุมมองต่อเศรษฐกิจ (Economic Sentiment)", -1.0, 1.0, 0.0, step=0.05
    )
    investor_confidence = st.slider(
        "ระดับความเชื่อมั่นในการลงทุน", 0.0, 100.0, 50.0, step=1.0
    )
    financial_stability = st.slider(
        "ดัชนีความเสถียรทางการเงินส่วนบุคคล", 0.3, 0.8, 0.6, step=0.01
    )

# ปุ่มคำนวณผล
if st.button("🚀 วิเคราะห์ผลและแนะนำอาชีพเสริม"):
    # 1. จัดเตรียมข้อมูล Input
    X_input = np.array([[
        monthly_income,
        monthly_expenditure,
        market_volatility,
        inflation_rate,
        investment_amount,
        savings_ratio,
        credit_score,
        debt_to_income,
        risk_tolerance,
        economic_sentiment,
        investor_confidence,
        financial_stability,
    ]])

    # 2. สร้าง Orange Table โดยส่งเฉพาะ Attributes (ไม่รวม Class) แล้วเติม Dummy Class
    if model.domain.class_vars:
        Y_dummy = np.array([[0.0]])
        input_orange_table = Orange.data.Table.from_numpy(
            model.domain, X_input, Y_dummy
        )
    else:
        input_orange_table = Orange.data.Table.from_numpy(
            model.domain, X_input
        )

    # 3. ทำนายผลด้วยโมเดล
    predictions = model(input_orange_table)
    pred_score = float(predictions[0])
    pred_score = np.clip(pred_score, 0, 100)

    st.subheader("📊 ผลการวิเคราะห์")
    st.metric(
        "คะแนนคำแนะนำทางการเงิน/การลงทุน (Recommendation Score)",
        f"{pred_score:.2f} / 100",
    )

    st.write("---")
    st.header("💡 อาชีพเสริมที่เหมาะสมกับระดับคะแนนของคุณ")

    if pred_score >= 70:
        st.success("🟢 **ระดับความพร้อมสูง (คะแนน 70 - 100):**")
        st.write(
            "- **ธุรกิจเปิดร้านค้าออนไลน์ / E-commerce:** เหมาะกับการลงทุนสต็อกสินค้าทำกำไรสูง"
        )
        st.write(
            "- **การลงทุนในสินทรัพย์ดิจิทัล / ตลาดทุน:** สำหรับผู้มีความพร้อมและรับความเสี่ยงได้สูง"
        )
        st.write("- **ซื้อแฟรนไชส์ขนาดเล็ก-กลาง:** ต่อยอดเงินออมและเครดิตที่มี")

    elif pred_score >= 45:
        st.info("🟡 **ระดับความพร้อมปานกลาง (คะแนน 45 - 69):**")
        st.write(
            "- **งานรับจ้างอิสระ (Freelance Skills):** กราฟิก, เขียนโปรแกรม, แปลภาษา"
        )
        st.write(
            "- **นายหน้าขายสินค้าออนไลน์ (Affiliate Marketing / Dropshipping):** ไม่ต้องใช้ทุนสูง"
        )
        st.write(
            "- **การสร้างดิจิทัลคอนเทนต์ / โค้ชชิ่งออนไลน์:** ดึงความเชี่ยวชาญเดิมมาทำเงิน"
        )

    else:
        st.warning("🔴 **ระดับเน้นความปลอดภัยและเซฟทุน (คะแนนน้อยกว่า 45):**")
        st.write(
            "- **งานบริการตามเวลาว่าง:** ขับรถรับส่ง / เดลิเวอรี่ / รับจ้างทั่วไป"
        )
        st.write(
            "- **ขายสินค้ามือสอง / สินค้าทำเอง (Handmade):** เน้นหมุนเวียนเงินโดยไม่เพิ่มหนี้สิน"
        )
