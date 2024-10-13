import os
from flask import Flask, request, jsonify
from keras.models import load_model
from tensorflow.keras.preprocessing import image  # เปลี่ยนที่นี่
import numpy as np

app = Flask(__name__)

# โหลดโมเดล
model_path = "C:/Users/chonn/OneDrive/Desktop/Streamlit/vehicle_classification_model_MobileNetV2.h5"
model = load_model(model_path)

# กำหนดพารามิเตอร์
IMG_SIZE = (150, 150)

@app.route('/')
def home():
    return "Welcome to the Vehicle Classification API!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # โหลดและประมวลผลภาพ
    img_path = os.path.join("uploads", file.filename)  # สร้างโฟลเดอร์สำหรับอัพโหลด
    file.save(img_path)  # บันทึกไฟล์ภาพ

    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # ปรับค่าช่องสีให้เป็น 0-1

    # ทำนาย
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions[0])  # หาค่าที่มีความน่าจะเป็นสูงสุด
    class_label = "Bike" if class_idx == 0 else "Car"  # สมมุติว่าคลาส 0 คือ Bike และ 1 คือ Car

    return jsonify({'class': class_label})

if __name__ == '__main__':
    app.run(host='0.0.0.0' , port = 5000, debug=False)
