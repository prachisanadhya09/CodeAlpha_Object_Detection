
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

st.set_page_config(page_title="Object Detection", page_icon="🔍")

st.title("AI Object Detection and Tracking")
st.write("Upload an image to detect objects using YOLO.")

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Detect Objects"):
        with st.spinner("Detecting objects..."):
            results = model.predict(np.array(image))

            result_image = results[0].plot()
            result_image = cv2.cvtColor(
                result_image, cv2.COLOR_BGR2RGB
            )

            st.subheader("Detection Result")
            st.image(result_image, use_container_width=True)

            detected = results[0].boxes
            st.write("Objects detected:", len(detected))

            for box in detected:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                name = model.names[class_id]
                st.write(
                    f"{name} - Confidence: {confidence:.2f}"
                )
