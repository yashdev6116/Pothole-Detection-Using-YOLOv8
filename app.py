import streamlit as st
import tempfile
import cv2
import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

import numpy as np
from PIL import Image
from ultralytics import YOLO
import time

# Load YOLOv8 model
model = YOLO("best_model_pothole.pt")

# Streamlit page settings
st.set_page_config(page_title="🚧 Pothole Detection App", layout="centered", page_icon="🛣️")

# Custom header
st.markdown("""
    <div style="text-align:center;">
        <h1 style="color:#FF4B4B;">🚧 Pothole Detection App 🛣️</h1>
        <p style="font-size:18px;">Upload an image or video to detect potholes using YOLOv8 🚀</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar for upload and settings
st.sidebar.header("⚙️ Settings")
file_type = st.sidebar.radio("Select file type", ("Image", "Video"))
uploaded_file = st.sidebar.file_uploader(f"Upload {file_type.lower()} file", type=["jpg", "jpeg", "png", "mp4", "mov"])
confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.4)

# Main app
if uploaded_file is not None:
    if file_type == "Image":
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="📸 Uploaded Image", use_column_width=True)

        image_np = np.array(image)
        with st.spinner("🔍 Detecting potholes in image..."):
            results = model.predict(source=image_np, conf=confidence)
            result_img = results[0].plot()

        st.image(result_img, caption="✅ Detection Result", use_column_width=True)

        # Show detection details
        st.subheader("📊 Detection Details")
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            st.write("❌ No potholes detected.")
        else:
            for i, box in enumerate(boxes):
                cls = model.names[int(box.cls[0])]
                conf = float(box.conf[0])
                st.success(f"✅ **Object {i+1}:** {cls} (Confidence: {conf:.2f})")

    elif file_type == "Video":
        # Save uploaded video to a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        cap = cv2.VideoCapture(tfile.name)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Prepare output video file
        out_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        stframe = st.empty()
        progress_bar = st.progress(0)
        frame_count = 0

        st.info("🎥 Processing video... please wait ⏳")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(source=frame, conf=confidence, verbose=False)
            result_frame = results[0].plot()

            out.write(result_frame)
            stframe.image(result_frame, channels="BGR", use_column_width=True)

            frame_count += 1
            progress = int((frame_count / total_frames) * 100)
            progress_bar.progress(min(progress, 100))

        cap.release()
        out.release()
        st.success("✅ Video processing complete!")

        st.video(out_path)

        # Download button
        with open(out_path, "rb") as file:
            video_bytes = file.read()
        st.download_button(
            label="⬇️ Download Processed Video",
            data=video_bytes,
            file_name="processed_pothole_video.mp4",
            mime="video/mp4"
        )

else:
    st.info("📥 Please upload an image or video to get started.")

# Custom footer
st.markdown("""
    <hr style="border-top: 2px solid #FF4B4B;">
    <div style="text-align:center; font-size:16px;">
        🚀 Made with ❤️ by <strong>YASH DEV</strong>
    </div>
""", unsafe_allow_html=True)
