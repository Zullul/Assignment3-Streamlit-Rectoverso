import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from tensorflow.keras.models import load_model
import requests
import urllib.request
import time

# ========== KONFIGURASI UBIDOTS ==========
UBIDOTS_TOKEN = 'BBUS-o6lQ26Q2Wbt2HriiXU7wgQvprs64QS'
DEVICE_NAME = 'ai-cam'

def send_to_ubidots(organik, anorganik):
    url = f"https://industrial.api.ubidots.com/api/v1.6/devices/{DEVICE_NAME}/"
    payload = {
        "organik": organik,
        "anorganik": anorganik
    }
    headers = {
        "X-Auth-Token": UBIDOTS_TOKEN,
        "Content-Type": "application/json"
    }
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        print("✅ Data berhasil dikirim ke Ubidots.")
    except requests.exceptions.RequestException as e:
        print(f"❌ Gagal mengirim ke Ubidots: {e}")

# ========== FUNGSI TAMBAHAN ==========
def get_mjpeg_frame(url):
    try:
        stream = urllib.request.urlopen(url)
        bytes_ = b''
        while True:
            bytes_ += stream.read(1024)
            a = bytes_.find(b'\xff\xd8')
            b = bytes_.find(b'\xff\xd9')
            if a != -1 and b != -1:
                jpg = bytes_[a:b+2]
                bytes_ = bytes_[b+2:]
                img_np = np.frombuffer(jpg, dtype=np.uint8)
                frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
                return frame
    except Exception as e:
        print(f"❌ Error parsing MJPEG: {e}")
        return None

@st.cache_resource
def load_yolo():
    return YOLO('yolov8n.pt')

@st.cache_resource
def load_classifier():
    return load_model('model_Rectoverso_best.h5')

def preprocess(img_cv):
    img_resized = cv2.resize(img_cv, (150, 150))
    img_array = img_resized / 255.0
    return np.expand_dims(img_array, axis=0)

def classify(img_cv):
    pred = clf_model.predict(preprocess(img_cv), verbose=0)
    return ('Anorganik', pred[0][0]) if pred[0][0] > 0.47 else ('Organik', pred[0][0])

# ========== LOAD MODEL ==========
yolo = load_yolo()
clf_model = load_classifier()

# ========== UI STREAMLIT ==========
st.title("📷 Deteksi Sampah: Kamera Lokal & ESP32-CAM")

mode = st.radio("Pilih Sumber Kamera", ["Kamera Lokal (Streamlit)", "ESP32-CAM"])

if mode == "Kamera Lokal (Streamlit)":
    enable = st.checkbox("Aktifkan Kamera")
    picture = st.camera_input("Ambil Gambar Sampah", disabled=not enable)

    if picture:
        img = Image.open(picture)
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # Jalankan deteksi dan klasifikasi
        results = yolo.predict(img_cv, conf=0.4, imgsz=416, verbose=False)
        annotated = img_cv.copy()
        organik_count, anorganik_count = 0, 0

        for r in results:
            for box, cls in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy()):
                if int(cls) == 0:
                    continue
                x1, y1, x2, y2 = map(int, box)
                crop = img_cv[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                label, score = classify(crop)
                color = (0, 255, 0) if label == 'Organik' else (255, 0, 0)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, f"{label} ({score:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                if label == 'Organik':
                    organik_count += 1
                else:
                    anorganik_count += 1

        st.image(annotated, channels="BGR", caption="Hasil Deteksi", use_container_width=True)
        st.success(f"🟢 Sampah Organik: {organik_count}")
        st.error(f"🔴 Sampah Anorganik: {anorganik_count}")
        send_to_ubidots(organik_count, anorganik_count)

# ========== MODE ESP32-CAM STREAM ==========
elif mode == "ESP32-CAM":
    esp32_url = st.text_input("Masukkan URL Stream ESP32-CAM", "http://192.168.1.28:81/stream")
    run = st.checkbox("Mulai Stream ESP32")
    frame_placeholder = st.empty()
    stop_button = st.button("STOP")

    if run and not stop_button:
        st.info("Stream berjalan...")

        while True:
            frame = get_mjpeg_frame(esp32_url)
            if frame is None:
                st.warning("Gagal membaca frame dari MJPEG stream.")
                break

            results = yolo.predict(frame, conf=0.4, imgsz=416, verbose=False)
            annotated = frame.copy()
            organik_count, anorganik_count = 0, 0

            for r in results:
                for box, cls in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy()):
                    if int(cls) == 0:
                        continue
                    x1, y1, x2, y2 = map(int, box)
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    label, score = classify(crop)
                    color = (0, 255, 0) if label == 'Organik' else (255, 0, 0)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated, f"{label} ({score:.2f})", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    if label == 'Organik':
                        organik_count += 1
                    else:
                        anorganik_count += 1

            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(annotated_rgb, channels="RGB", use_column_width=True)

            send_to_ubidots(organik_count, anorganik_count)

            time.sleep(0.5)

            if stop_button:
                st.success("Stream dihentikan.")
                break