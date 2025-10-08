# app.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(page_title="Coastal Change Detection POC", layout="wide")
st.title("🌊 Coastal Change Detection — Data Lake POC")

st.write("Upload two satellite images of the same coastal area (different years) to detect changes.")

# --- File upload ---
img1_file = st.file_uploader("Upload First Image (Earlier Year)", type=["jpg", "jpeg", "png"])
img2_file = st.file_uploader("Upload Second Image (Recent Year)", type=["jpg", "jpeg", "png"])

if img1_file and img2_file:
    # Load images as numpy arrays
    img1 = np.array(Image.open(img1_file))
    img2 = np.array(Image.open(img2_file))

    # Resize both images to same dimensions
    img1 = cv2.resize(img1, (800, 800))
    img2 = cv2.resize(img2, (800, 800))

    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    # --- Change Detection ---
    diff = cv2.absdiff(gray1, gray2)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    changed = cv2.countNonZero(thresh)
    total = thresh.size
    percent_change = round((changed / total) * 100, 2)

    # --- Safe Highlight ---
    # Ensure img2 is RGB
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB) if img2.shape[-1] == 3 else img2

    # Convert grayscale mask to 3 channels
    mask_rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    mask_rgb = cv2.resize(mask_rgb, (img2_rgb.shape[1], img2_rgb.shape[0]))

    # Apply red overlay where changes detected
    highlight = img2_rgb.copy()
    highlight[mask_rgb[:, :, 0] > 0] = [255, 0, 0]

    # --- Display Results ---
    st.subheader(f"Change Detected: {percent_change}%")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(img1, caption="Earlier Image", use_column_width=True)
    with col2:
        st.image(img2, caption="Recent Image", use_column_width=True)
    with col3:
        st.image(highlight, caption="Detected Changes (Red Areas)", use_column_width=True)

    # --- CSV download ---
    csv_data = f"Location,Year1,Year2,PercentChange\nVizag,2021,2024,{percent_change}\n"
    st.download_button("Download Change Summary (CSV)", csv_data, file_name="change_summary.csv")

else:
    st.info("Please upload both images to start the change detection process.")
