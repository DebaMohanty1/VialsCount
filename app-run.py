import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import base64
from datetime import datetime
from ultralytics import YOLO

# Streamlit app styling
st.set_page_config(
    page_title="Medical Equipments Detection Streamlit App",
    page_icon="👁️",  # Eye icon for object detection
    layout='wide'
)

# Function to convert image to bytes
def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

# Load the model
model_path = "Yolov8n_30.pt"
try:
    model = YOLO(model_path)
except Exception as e:
    st.write(f"Error loading model: {e}")

# Function to make predictions and draw bounding boxes with midpoints
def predict_and_draw(image, conf_threshold=0.70, iou_threshold=0.45):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    try:
        results = model.predict(source=image, conf=conf_threshold, iou=iou_threshold, verbose=False, max_det=1000, save=False)
        if not results or not results[0].boxes.data.size(0):
            st.write("No detections found.")
            return image, {}, datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        labels, coords = results[0].boxes.data[:, -1], results[0].boxes.data[:, :-1]

        img_draw = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_draw)

        # Dynamically adjust font size based on image size
        width, height = img_draw.size
        font_size = max(10, width // 50)  # Adjust font size as needed
        
        try:
            # Try loading the font
            font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
            font = ImageFont.truetype(font_path, font_size)
        except IOError:
            # Fallback to default font if the specific font cannot be loaded
            font = ImageFont.load_default()
        
        class_counts = {}

        # Sort detections by top-left corner (x1, y1) coordinates
        sorted_detections = sorted(zip(labels, coords), key=lambda x: (x[1][0], x[1][1]))

        for i, (label, coord) in enumerate(sorted_detections):
            x1, y1, x2, y2, conf = coord
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Draw bounding box
            draw.rectangle(((x1, y1), (x2, y2)), outline='red', width=2)

            # Draw midpoint
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            draw.ellipse((mid_x-2, mid_y-2, mid_x+2, mid_y+2), fill='red', outline='red')
            draw.text((mid_x + 4, mid_y - 4), str(i + 1), fill="white", font=font)  # Add unique number to each vial

            label = int(label.item())
            if label == 0:
                class_name = "Vials"
            elif label == 1:
                class_name = "PFS"
            else:
                class_name = "Unknown"
            
            if class_name in class_counts:
                class_counts[class_name] += 1
            else:
                class_counts[class_name] = 1

        # Add total count based on class
        y_offset = 10  # Starting y position for text
        for class_name, count in class_counts.items():
            text = f"{class_name}: {count}"
            draw.text((10, y_offset), text, fill="white", font=font)
            y_offset += font_size + 5  # Increase y position for the next class count

        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        draw.text((10, y_offset), f"Timestamp: {timestamp}", fill="white", font=font)

        return img_draw, class_counts
    except Exception as e:
        st.write(f"An error occurred during prediction: {e}")
        return image, {}

# Clear cache
st.cache_data.clear()

header_html = "<img src='data:image/png;base64,{}' class='img-fluid' style='width: 100px; height: auto;'>".format(
    img_to_bytes("aispry.webp")
)
header2_html = "<img src='data:image/png;base64,{}' class='img-fluid' style='width: 150px; height: auto;'>".format(
    img_to_bytes("Dr-Reddys-Laboratories.webp")
)

# Streamlit app title and header
title = "Medical Vials Counting"

html_temp = f"""
    <div style="display: flex; align-items: center; justify-content: center; background: linear-gradient(90deg, #66a3ff, #ff3333); border-radius: 25px; padding: 10px; color: black; text-align: center; margin: 5px 0px; box-shadow: 5px 5px 15px 5px #000000;">
        <div style="flex: 1; display: flex; justify-content: flex-start;">
            {header_html}
        </div>
        <div style="flex: 2; text-align: center;">
            <p style="font-size: 25px; font-weight: bold; margin: 0;">
                {title}
            </p>
        </div>
        <div style="flex: 1; display: flex; justify-content: flex-end;">
            {header2_html}
        </div>
    </div>
"""

# Display the header
st.markdown(html_temp, unsafe_allow_html=True)

# Sidebar for file upload and camera capture
st.sidebar.header("Upload or Capture Image")

# Options to either upload a file or use camera
option = st.sidebar.radio("Select Input Method", ("Upload File", "Use Camera"))

# Process uploaded file or camera capture
image = None
if option == "Upload File":
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
elif option == "Use Camera":
    st.sidebar.write("Please press the 'Start Camera' button to capture an image.")
    image = st.camera_input("Take a photo")
    if image is not None:
        image = Image.open(image)

if image is not None:
    processed_image, class_counts = predict_and_draw(image)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(image, caption='Original Image', use_column_width=True)
    with col2:
        st.subheader("Processed Image")
        st.image(processed_image, caption='Processed Image with Annotations', use_column_width=True)
    
    # Display class counts
    st.write("Detected items with counts:")
    for class_name, count in class_counts.items():
        st.write(f"{class_name}: {count}")
else:
    st.write("Upload an image or use the camera to start processing.")
