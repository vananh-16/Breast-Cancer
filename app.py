import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

st.set_page_config(page_title="Breast Cancer Classification", layout="centered")

st.title("Breast Cancer Detection Demo")

st.write("""Ứng dụng này sử dụng mô hình CNN đã được huấn luyện để dự đoán xem một ảnh có dấu hiệu ung thư hay không. Hãy upload một ảnh để tiến hành dự đoán.""")

# Load model
@st.cache_resource
def load_cnn_model():
	model = load_model('breast_cancer_model.keras')
	return model

model = load_cnn_model()

uploaded_file = st.file_uploader("Chọn một file ảnh (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Đọc file ảnh từ bộ nhớ
	img = Image.open(uploaded_file).convert('RGB')
 # Hiển thị ảnh gốc
st.write("Ảnh đã upload:")
st.image(img, use_column_width=True)

    # Tiền xử lý ảnh cho model
img_resized = img.resize((224, 224))
img_array = image.img_to_array(img_resized)/255.0
img_array = np.expand_dims(img_array, axis=0)

    # Dự đoán
prediction = model.predict(img_array)[0][0]

    # Ngưỡng: nếu >=0.5 là "Cancer: Yes", ngược lại "Cancer: No"
threshold = 0.5
if prediction >= threshold:
	label = "Cancer: Yes"
	color = (255, 0, 0)  # Red
else:
	label = "Cancer: No"
	color = (0, 255, 0)  # Green

    # Chuyển ảnh sang OpenCV format để vẽ rectangle
img_cv = np.array(img)
img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

    # Vẽ một bounding box lên ảnh (đơn giản là khung viền quanh ảnh)
height, width, _ = img_cv.shape
padding = 50
start_point = (padding, padding)
img_with_rectangle = cv2.rectangle(img_cv.copy(), start_point, end_point, color, 5)
 # Vẽ label
img_with_text = cv2.putText(img_with_rectangle, label, (50, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

    # Chuyển BGR -> RGB để hiển thị trên Streamlit
img_rgb = cv2.cvtColor(img_with_text, cv2.COLOR_BGR2RGB)

st.write("Kết quả dự đoán:")
st.image(img_rgb, use_column_width=True)
st.write(f"Dự đoán: {label} (Giá trị dự đoán: {prediction:.4f})")
