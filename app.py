import streamlit as st
import cv2
import numpy as np
import easyocr
from ultralytics import YOLO
import os
import csv
from datetime import datetime
from PIL import Image
import pandas as pd

# ------------------ PATH SETUP ------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(BASE_DIR, "plate_output.csv")
SNAPSHOT_DIR = os.path.join(BASE_DIR, "snapshots")
STATIC_DIR = os.path.join(BASE_DIR, "static")
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")

os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# Create CSV if not exists
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["vehicle", "translated_plate_number_nepali", "translated_plate_number_english", "snapshot"])

# ------------------ LOAD MODELS ------------------
plate_model = YOLO(MODEL_PATH)  # YOLO model
reader = easyocr.Reader(['ne'])  # Nepali OCR

# ------------------ TRANSLITERATION MAP ------------------
transliteration_map = {
    "क": "ka", "को": "ko", "ख": "kha", "ग": "ga", "च": "cha", "ज": "ja", "झ": "jha", "ञ": "ya",
    "डि": "di", "त": "ta", "ना": "na", "प": "pa", "प्र": "pra", "ब": "ba", "बा": "baa",
    "भे": "bhe", "म": "ma", "मे": "me", "य": "ya", "लु": "lu", "सी": "si", "सु": "su",
    "से": "se", "ह": "ha", "०": "0", "१": "1", "२": "2", "३": "3", "४": "4",
    "५": "5", "६": "6", "७": "7", "८": "8", "९": "9"
}

# ------------------ DETECTION FUNCTION ------------------
def detect_and_read(frame_rgb):
    results = reader.readtext(frame_rgb)
    full_nepali, full_latin = [], []
    annotated_image = frame_rgb.copy()

    for (bbox, text, _) in results:
        full_nepali.append(text)
        translated = ''.join([transliteration_map.get(ch, ch) for ch in text])
        full_latin.append(translated)

        pts = np.array(bbox).astype(int)
        cv2.polylines(annotated_image, [pts], True, (0, 255, 0), 2)
        cv2.putText(annotated_image, translated, tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return [(" ".join(full_nepali), " ".join(full_latin), (0, 0))], cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

# ------------------ VEHICLE PROFILE ------------------
def get_vehicle_profile(nep_text):
    vehicle_type, zone, plate_type = "N/A", "N/A", "Unknown"
    img_path = None

    if "ज" in nep_text:
        vehicle_type, plate_type = "Taxi", "Public"
        img_path = os.path.join(STATIC_DIR, "taxi.png")
    elif "च" in nep_text:
        vehicle_type, plate_type = "Car", "Private"
        img_path = os.path.join(STATIC_DIR, "car.png")
    elif "प" in nep_text:
        vehicle_type, plate_type = "2 Wheeler", "Private"
        img_path = os.path.join(STATIC_DIR, "bike.png")

    if "बा" in nep_text or "प्रदेश ३" in nep_text:
        zone = "Bagmati"
    elif "लु" in nep_text:
        zone = "Lumbini"
    elif "ग" in nep_text:
        zone = "Gandaki"

    return vehicle_type, zone, img_path, plate_type

# ------------------ IMAGE PROCESSING ------------------
def process_image(image_bytes, vehicle_count):
    image = Image.open(image_bytes).convert("RGB")
    frame_rgb = np.array(image)

    results, annotated_img = detect_and_read(frame_rgb)

    if results:
        full_nepali, full_latin = "", ""
        for nep, lat, _ in results:
            full_nepali += nep
            full_latin += lat

        vehicle_type, zone, img_path, plate_type = get_vehicle_profile(full_nepali)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_path = os.path.join(SNAPSHOT_DIR, f"plate_{vehicle_count}_{timestamp}.jpg")
        cv2.imwrite(snapshot_path, annotated_img)

        with open(CSV_FILE, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([vehicle_count, full_nepali, full_latin, snapshot_path])

        return vehicle_count, full_nepali, full_latin, snapshot_path, annotated_img, plate_type, vehicle_type, zone, img_path

    return None, None, None, None, frame_rgb, "Unknown", "N/A", "N/A", None

# ------------------ STREAMLIT UI ------------------
st.set_page_config(page_title="Nepali Plate Detector", layout="centered")

st.title("🇳🇵 Nepali Vehicle Number Plate Recognition")
st.markdown("Upload an image to detect and read Nepali number plates.")

uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    vehicle_id = sum(1 for _ in open(CSV_FILE))  # row count
    vehicle_id = max(1, vehicle_id)

    with st.spinner("🔍 Processing image..."):
        vid, nep, lat, snapshot, processed_img, plate_type, vehicle_type, zone, img_path = process_image(uploaded_file, vehicle_id)

    if nep:
        st.success("✅ Number plate detected!")
        st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), caption="📸 Snapshot", use_container_width=True)

        st.subheader("🔎 Extracted Plate Details")
        st.table({
            "Vehicle ID": [vid],
            "Plate (Nepali)": [nep],
            "Plate (English)": [lat],
            "Snapshot Path": [snapshot]
        })

        if st.button("🚘 Show Vehicle Profile"):
            st.markdown("---")
            st.write(f"**Type of Vehicle:** {vehicle_type}")
            st.write(f"**Zonal Representation:** {zone}")
            st.write(f"**Plate Type:** {plate_type}")
            if img_path and os.path.exists(img_path):
                st.image(img_path, caption=vehicle_type, use_container_width=False)

    else:
        st.error("❌ Could not detect a number plate.")

if st.checkbox("📄 Show All Detected Plates"):
    df = pd.read_csv(CSV_FILE)
    st.dataframe(df)
