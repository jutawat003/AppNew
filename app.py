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
    "ระบบประเมินจากโมเดล Linear Regression ร่วมกับการวิเคราะห์ความถนัดส่วนบุคคล"
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
st.header("1. กรอกข้อมูลทางการเงินและความถนัดของคุณ")

col1, col2 = st.columns(2)

with col1:
    monthly_income = st.number_input(
        "รายได้ต่อเดือน (บาท)", value=30000.0, step=1000.0
    )

with col2:
    monthly_expenditure = st.number_input(
        "ค่าใช้จ่ายต่อเดือน (บาท)", value=18000.0, step=1000.0
    )

# ส่วนเลือกทักษะ/ความถนัด
user_skills = st.multiselect(
    "🎯 เลือกทักษะ หรือความถนัดที่คุณมี (เลือกได้มากกว่า 1 ข้อ):",
    [
        "การขาย / การตลาดออนไลน์",
        "ออกแบบ / ออกแบบกราฟิก / วาดรูป",
        "การเขียนโปรแกรม / ทำเว็บ",
        "ทำอาหาร / ทำขนม",
        "ภาษาต่างประเทศ / แปลภาษา",
        "ถ่ายภาพ / ตัดต่อวิดีโอ",
        "ขับรถ / รู้เส้นทาง",
        "สอนหนังสือ / ติวเตอร์",
        "งานฝีมือ / Handmade",
        "ไม่มีทักษะเฉพาะ (เน้นงานใช้เวลาว่าง)",
    ],
    default=["การขาย / การตลาดออนไลน์"],
)

# ปุ่มคำนวณผล
if st.button("🚀 วิเคราะห์ผลและแนะนำอาชีพเสริม"):
    # คำนวณค่าตัวแปรอื่นๆ อัตโนมัติจากรายได้และรายจ่าย
    net_savings = max(0.0, monthly_income - monthly_expenditure)
    savings_ratio = net_savings / monthly_income if monthly_income > 0 else 0.0
    savings_ratio = min(0.6, savings_ratio)

    investment_amount = net_savings * 3
    debt_to_income = min(0.7, monthly_expenditure / monthly_income)

    # ค่ามาตรฐานกลางๆ
    market_volatility = 30.0
    inflation_rate = 4.0
    credit_score = 700.0
    risk_tolerance = 0.5
    economic_sentiment = 0.0
    investor_confidence = 50.0
    financial_stability = 0.6

    # 1. จัดเตรียมข้อมูล Input 12 ค่า
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

    # 2. สร้าง Orange Table
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

    st.subheader("📊 ผลการวิเคราะห์สภาพทางการเงิน")
    st.metric(
        "คะแนนความพร้อมทางการเงิน (Recommendation Score)",
        f"{pred_score:.2f} / 100",
    )

    # คำอธิบายคะแนน
    with st.expander("❓ คะแนนนี้หมายความว่าอย่างไร?", expanded=True):
        if pred_score >= 70:
            st.write(
                f"👉 **คะแนน {pred_score:.2f} (ความพร้อมสูง):** สภาพคล่องทางการเงินและเงินออมดีเยี่ยม มีความพร้อมในการนำเงินทุนไปลงทุนต่อยอดเพื่อสร้างอาชีพเสริมที่ใช้เงินก้อนได้"
            )
        elif pred_score >= 45:
            st.write(
                f"👉 **คะแนน {pred_score:.2f} (ความพร้อมปานกลาง):** สภาพคล่องปานกลาง เหมาะกับการทำอาชีพเสริมที่เน้นใช้ทักษะความสามารถ โดยไม่ต้องใช้เงินก้อนใหญ่เพื่อลดความเสี่ยง"
            )
        else:
            st.write(
                f"👉 **คะแนน {pred_score:.2f} (เน้นความปลอดภัย):** ควรเน้นรักษาเงินต้น เลือกอาชีพเสริมที่ไม่ต้องลงทุนเงินสด เพื่อสร้างรายได้เพิ่มโดยไม่เพิ่มภาระหนี้สิน"
            )

    st.write("---")
    st.header("💡 อาชีพเสริมที่แนะนำเฉพาะคุณ (คัดสรรจากทักษะ + ศักยภาพเงิน)")

    recommendations = []

    for skill in user_skills:
        if skill == "การขาย / การตลาดออนไลน์":
            if pred_score >= 70:
                recommendations.append(
                    "🛍️ **นำเข้าสินค้ามาขายออนไลน์ / สต็อกสินค้าเอง:** นำเงินออมบางส่วนมาลงทุนสต็อกสินค้าเพื่อสร้างกำไรต่อชิ้นสูงขึ้น"
                )
            else:
                recommendations.append(
                    "📱 **ทำ Affiliate Marketing / Dropshipping:** ตัวแทนขายสินค้าออนไลน์โดยไม่ต้องสต็อกสินค้าและไม่ต้องใช้เงินทุน"
                )

        elif skill == "ออกแบบ / ออกแบบกราฟิก / วาดรูป":
            recommendations.append(
                "🎨 **รับทำกราฟิก / วาดสติ๊กเกอร์ Line / ขายภาพสต็อก:** ใช้ทักษะสร้างรายได้ผ่านเว็บ Fastwork, Fiverr หรือขายสติ๊กเกอร์"
            )

        elif skill == "การเขียนโปรแกรม / ทำเว็บ":
            recommendations.append(
                "💻 **รับพัฒนาเว็บไซต์ / แอพพลิเคชัน ฟรีแลนซ์:** ตลาดมีความต้องการสูง รายได้ต่อชิ้นดี ไม่ต้องใช้เงินทุนเริ่มต้น"
            )

        elif skill == "ทำอาหาร / ทำขนม":
            if pred_score >= 70:
                recommendations.append(
                    "🍲 **เปิดร้านอาหาร / เบเกอรี่สั่งทำ (Pre-order) หรือเดลิเวอรี่:** ลงทุนอุปกรณ์เพิ่มเติมเพื่อสร้างแบรนด์ของตัวเอง"
                )
            else:
                recommendations.append(
                    "🍪 **ทำขนม/อาหารว่างขายตามออเดอร์:** เน้นรับเงินมัดจำล่วงหน้าเพื่อนำมาซื้อวัตถุดิบ ลดความเสี่ยงเงินจม"
                )

        elif skill == "ภาษาต่างประเทศ / แปลภาษา":
            recommendations.append(
                "🗣️ **รับแปลเอกสาร / สอนภาษาออนไลน์ / ไกด์ท้องถิ่น:** สร้างรายได้จากทักษะภาษาผ่านแพลตฟอร์มออนไลน์"
            )

        elif skill == "ถ่ายภาพ / ตัดต่อวิดีโอ":
            recommendations.append(
                "📸 **รับถ่ายภาพงานต่างๆ / ตัดต่อ คลิป Short, Reel, TikTok:** ตลาดการทำคลิปสั้นกำลังเติบโตสูงมาก"
            )

        elif skill == "ขับรถ / รู้เส้นทาง":
            recommendations.append(
                "🚗 **ขับรถรับส่งผู้โดยสาร / เดลิเวอรี่ส่งของ (Grab, Lalamove):** ใช้เวลาว่างหลังเลิกงานสร้างรายได้ทันที"
            )

        elif skill == "สอนหนังสือ / ติวเตอร์":
            recommendations.append(
                "📚 **รับสอนพิเศษออนไลน์ / ทำคอร์สเรียนออนไลน์ขาย:** ลงทุนเวลาสร้างคอร์สเพียงครั้งเดียว ขายสร้าง Passive Income ได้เรื่อยๆ"
            )

        elif skill == "งานฝีมือ / Handmade":
            recommendations.append(
                "🧶 **ทำสินค้า Handmade ขายออนไลน์:** เช่น งานถัก, เครื่องประดับ, ของตกแต่งบ้าน ขายใน Etsy หรือ Shopee"
            )

        elif skill == "ไม่มีทักษะเฉพาะ (เน้นงานใช้เวลาว่าง)":
            if pred_score >= 70:
                recommendations.append(
                    "🏦 **ลงทุนในกองทุนรวม / สินทรัพย์ปันผล:** ใช้เงินออมทำงานแทนผ่านการลงทุนความเสี่ยงต่ำ-ปานกลาง"
                )
            else:
                recommendations.append(
                    "📦 **รับจ้างแพ็คของ / ขับเดลิเวอรี่ / ขายสินค้ามือสองที่มีอยู่:** เป็นการเริ่มต้นสร้างรายได้ที่ง่ายที่สุดโดยไม่ต้องใช้เงินทุน"
                )

    if recommendations:
        for rec in recommendations:
            st.info(rec)
    else:
        st.write("กรุณาเลือกความถนัดอย่างน้อย 1 อย่างเพื่อรับคำแนะนำครับ")
