import streamlit as st
import tflite_runtime.interpreter as tflite
from PIL import Image, ImageOps
import numpy as np

# ==========================================
# 1. ตั้งค่าหน้าเว็บ GUI แบบกว้าง
# ==========================================
st.set_page_config(page_title="AI คัดแยกขยะ (กลุ่ม 7)", layout="wide")
st.title("♻️ AI ผู้ช่วยคัดแยกขยะอัจฉริยะ (กลุ่ม 7)")
st.write("เปิดกล้องเว็บแคมแล้วชูขยะให้ AI ช่วยวิเคราะห์ประเภทขยะได้เลยครับ")

# ==========================================
# 2. ฟังก์ชันโหลดโมเดลแบบ TF Lite (ไม่ใช้ cache เพื่อเซฟ RAM บัญชีใหม่)
# ==========================================
def load_tflite_model():
    interpreter = tflite.Interpreter(model_path="model_unquant.tflite")
    interpreter.allocate_tensors()
    return interpreter

def load_our_labels():
    with open("labels.txt", "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f.readlines()]
    return labels

try:
    interpreter = load_tflite_model()
    labels = load_our_labels()
    
    # ดึงข้อมูล Input/Output Details ของโมเดล
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
except Exception as e:
    st.error(f"⚠️ มีบางอย่างผิดพลาด กรุณาเช็คว่ามีไฟล์ model_unquant.tflite และ labels.txt อยู่ในโฟลเดอร์เดียวกันหรือยังครับ: {e}")
    st.stop()

# ==========================================
# 3. จัดวางหน้าจอแบ่ง ซ้าย-ขวา
# ==========================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("📷 กล้องตรวจสอบขยะ")
    camera_image = st.camera_input("วางขยะให้อยู่ในกรอบกล้อง แล้วกดปุ่ม Take Photo")

with col2:
    st.subheader("📊 ผลลัพธ์การวิเคราะห์")
    
    if camera_image is not None:
        image = Image.open(camera_image)
        
        with st.spinner("🔄 AI กำลังสแกนหาประเภทขยะ..."):
            # เตรียมรูปภาพ 224x224
            size = (224, 224)
            image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
            image_array = np.asarray(image)
            
            if image_array.shape[-1] == 4:
                image_array = image_array[..., :3]
                
            normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
            data = np.expand_dims(normalized_image_array, axis=0)

            # --- ประมวลผลด้วย TF Lite ---
            interpreter.set_tensor(input_details[0]['index'], data)
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]['index'])
            
            index = np.argmax(prediction)
            class_name = labels[index]
            confidence_score = prediction[0][index]

        # ==========================================
        # 4. แสดงผลแยกสีตามประเภทขยะ
        # ==========================================
        st.write(f"📊 **ความแม่นยำในการตรวจจับ:** {confidence_score * 100:.2f}%")
        
        if "อันตราย" in class_name:
            st.error("🚨 ขยะอันตราย (Hazardous Waste)")
            st.write("📌 **วิธีจัดการ:** กรุณาทิ้งลง **ถังสีแดง** (เช่น ถ่านไฟฉาย, หลอดไฟ, กระป๋องสเปรย์)")
        elif "อินทรีย์" in class_name or "เปียก" in class_name:
            st.success("🌱 ขยะอินทรีย์ (Organic Waste)")
            st.write("📌 **วิธีจัดการ:** กรุณาทิ้งลง **ถังสีเขียว** (เช่น เศษอาหาร, เปลือกผลไม้, เศษผัก)")
        elif "รีไซเคิล" in class_name:
            st.warning("♻️ ขยะรีไซเคิล (Recyclable Waste)") 
            st.write("📌 **วิธีจัดการ:** กรุณาทิ้งลง **ถังสีเหลือง** (เช่น ขวดพลาสติก, แก้ว, กระดาษ)")
        elif "ทั่วไป" in class_name:
            st.info("🗑️ ขยะทั่วไป (General Waste)") 
            st.write("📌 **วิธีจัดการ:** กรุณาทิ้งลง **ถังสีน้ำเงิน** (เช่น ซองขนม, กล่องโฟมเปื้อนอาหาร)")
        else:
            clean_name = class_name[2:] if class_name[0].isdigit() else class_name
            st.success(f"🔍 ตรวจพบ: **{clean_name}**")
