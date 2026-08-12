import streamlit as st
from PIL import Image
import os
import random
import pandas as pd
from datetime import datetime
import numpy as np

# ================== PAGE CONFIG ==================
st.set_page_config(page_title="PV Fault Dashboard", layout="wide")
st.title("⚡ PV Fault Detection Dashboard")

# ================== SIDEBAR ==================
st.sidebar.header("⚙️ Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)

# ================== FOLDERS ==================
os.makedirs("data/uploads", exist_ok=True)
os.makedirs("data/low_conf", exist_ok=True)

import os

model_available = False

try:
    from ultralytics import YOLO

    st.write("Current files:", os.listdir())
    
    if os.path.exists("model"):
        st.write("Model folder found:", os.listdir("model"))
    
    if os.path.exists("model/best.pt"):
        model = YOLO("model/best.pt")
        model_available = True
        st.success("Model loaded successfully ✅")
    else:
        st.error("best.pt NOT found ❌")

except Exception as e:
    st.error(f"Model loading failed: {e}")
    model_available = False

# ================== FILE UPLOAD ==================
file = st.file_uploader("📤 Upload PV Image", type=["jpg", "png", "jpeg"])

if file is None:
    st.info("👆 Please upload a PV image to start detection")

# ================== MAIN LOGIC ==================
if file:
    img = Image.open(file)

    col1, col2 = st.columns(2)

    # -------- ORIGINAL IMAGE --------
    with col1:
        st.subheader("📸 Original Image")
        st.image(img, use_container_width=True)

    # Save image
    img_path = f"data/uploads/{file.name}"
    img.save(img_path)

    # ================== DETECTION ==================
    with col2:
        st.subheader("🔍 Detection Results")

        fault_list = []
        total_faults = 0

        if model_available:
            results = model.predict(img, conf=confidence_threshold)

            for r in results:
                plotted_img = r.plot()

                boxes = r.boxes
                if boxes is not None:
                    total_faults += len(boxes)

                    for b in boxes:
                        cls = int(b.cls[0])
                        name = model.names[cls]
                        conf = float(b.conf[0])

                        fault_list.append(name)

                        # ✅ Confidence Display
                        st.write(f"🔍 {name} - Confidence: {conf:.2f}")

            # Show detection image
            st.image(plotted_img, caption="Detected Image", use_container_width=True)

            # ✅ Download Button
            img_bytes = cv2.imencode('.png', plotted_img)[1].tobytes()
            st.download_button(
                label="📥 Download Result Image",
                data=img_bytes,
                file_name="detected_image.png",
                mime="image/png"
            )

            st.success("Model Detection Active ✅")

        else:
            # -------- DUMMY MODE --------
            fault_types = ["crack", "dust", "hotspot"]
            fault_list = random.choices(fault_types, k=random.randint(1, 3))
            total_faults = len(fault_list)

            st.warning("Using Dummy Detection (Model not loaded)")

        # ================== METRICS ==================
        st.metric("Total Faults", total_faults)

        if fault_list:
            st.write("Detected Faults:", fault_list)

        # ✅ Fault Severity
        if total_faults <= 2:
            severity = "🟢 Low"
        elif total_faults <= 5:
            severity = "🟡 Medium"
        else:
            severity = "🔴 High"

        st.write(f"⚠️ Fault Severity Level: {severity}")

        # ================== ALERT SYSTEM ==================
        if total_faults >= 5:
            st.error("🚨 High Fault Detected! Immediate Maintenance Required")
        elif total_faults >= 3:
            st.warning("⚠️ Moderate Fault Level")
        else:
            st.success("✅ System is Healthy")

        # ================== EFFICIENCY ==================
        efficiency = max(0, 100 - (total_faults * 5))
        st.metric("⚡ Estimated Efficiency (%)", efficiency)

        # ================== ACTIVE LEARNING ==================
if model_available:
    for r in results:
        if r.boxes is not None and len(r.boxes) > 0:
            avg_conf = float(r.boxes.conf.mean())

            # Show confidence (for debugging)
            st.write(f"Average Confidence: {avg_conf:.2f}")

            # Force save if confidence is below 0.9 (easier trigger)
            if avg_conf < 0.9:
                img.save(f"data/low_conf/{file.name}")
                st.warning("Saved to low_conf for retraining ⚠️")

    # ================== CHARTS ==================
    st.subheader("📊 Fault Distribution")

    if fault_list:
        df = pd.DataFrame(fault_list, columns=["Fault"])
        chart_data = df["Fault"].value_counts().reset_index()
        chart_data.columns = ["Fault", "Count"]

        st.bar_chart(chart_data.set_index("Fault"))

        st.subheader("📊 Fault Details Table")
        st.dataframe(chart_data)

    # ================== LOGGING ==================
    log_file = "data/logs.csv"

    log_data = {
        "image": file.name,
        "faults": ",".join(fault_list),
        "count": total_faults,
        "efficiency": efficiency,
        "time": datetime.now()
    }

    df_log = pd.DataFrame([log_data])
    df_log.to_csv(log_file, mode='a', header=not os.path.exists(log_file), index=False)

# ================== LOG HISTORY ==================
st.subheader("📁 Previous Logs")

if os.path.exists("data/logs.csv"):
    df_logs = pd.read_csv("data/logs.csv")
    st.dataframe(df_logs)
else:
    st.info("No logs available yet.")
