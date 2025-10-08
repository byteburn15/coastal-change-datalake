
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

st.title("🌊 Coastal Change Detection (POC)")

st.write("Upload two satellite images of the same coastal area (different years) to detect and visualize changes.")

# Upload two images
img1_file = st.file_uploader("Upload First Image (Earlier Year)", type=["jpg", "jpeg", "png"])
img2_file = st.file_uploader("Upload Second Image (Recent Year)", type=["jpg", "jpeg", "png"])

if img1_file and img2_file:
    img1 = np.array(Image.open(img1_file))
    img2 = np.array(Image.open(img2_file))

    # Resize both
    img1 = cv2.resize(img1, (800, 800))
    img2 = cv2.resize(img2, (800, 800))

    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    # Change detection
    diff = cv2.absdiff(gray1, gray2)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    changed = cv2.countNonZero(thresh)
    total = thresh.size
    percent_change = round((changed / total) * 100, 2)

    # --- Highlight changes safely ---
    mask = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    highlight = np.where(mask == 255, [255, 0, 0], img2)

    st.subheader(f"Change Detected: {percent_change}%")

    col1, col2, col3 = st.columns(3)
    with col1: st.image(img1, caption="Earlier Image")
    with col2: st.image(img2, caption="Recent Image")
    with col3: st.image(highlight, caption="Detected Changes")

    # Highlight changes
    highlight = img2.copy()
    mask = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    highlight = np.where(mask == 255, [255, 0, 0], highlight)

    st.subheader(f"Change Detected: {percent_change}%")

    col1, col2, col3 = st.columns(3)
    with col1: st.image(img1, caption="Earlier Image")
    with col2: st.image(img2, caption="Recent Image")
    with col3: st.image(highlight, caption="Detected Changes")

    csv_data = f"Location,Year1,Year2,PercentChange\nVizag,2021,2024,{percent_change}\n"
    st.download_button("Download Change Summary (CSV)", csv_data, file_name="change_summary.csv")
