import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Coastal Change Detection POC", layout="wide")
st.title("🌊 Coastal Change Detection — Data Lake POC")

st.write("Upload two satellite images of the same coastal area (different years) to detect changes.")

# --- File upload ---
img1_file = st.file_uploader("Upload First Image (Earlier Year)", type=["jpg", "jpeg", "png"])
img2_file = st.file_uploader("Upload Second Image (Recent Year)", type=["jpg", "jpeg", "png"])

if img1_file and img2_file:
    # Load images
    img1 = np.array(Image.open(img1_file))
    img2 = np.array(Image.open(img2_file))

    # Resize
    img1 = cv2.resize(img1, (800, 800))
    img2 = cv2.resize(img2, (800, 800))

    # Grayscale conversion
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    # Change detection
    diff = cv2.absdiff(gray1, gray2)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    changed = cv2.countNonZero(thresh)
    total = thresh.size
    percent_change = round((changed / total) * 100, 2)

    # --- Safe Highlight using cv2 ---
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB) if img2.shape[-1] == 3 else img2
    highlight = img2_rgb.copy()
    mask = thresh.astype(np.uint8)

    red = np.zeros_like(img2_rgb)
    red[:, :, 0] = 255  # R
    red[:, :, 1] = 0    # G
    red[:, :, 2] = 0    # B

    highlight = np.where(mask[:, :, None] == 255, red, highlight)

    # --- Display ---
    st.subheader(f"Change Detected: {percent_change}%")
    col1, col2, col3 = st.columns(3)
    with col1: st.image(img1, caption="Earlier Image", use_container_width=True)
    with col2: st.image(img2, caption="Recent Image", use_container_width=True)
    with col3: st.image(highlight, caption="Detected Changes (Red Areas)", use_container_width=True)

    # CSV download
    csv_data = f"Location,Year1,Year2,PercentChange\nVizag,2021,2024,{percent_change}\n"
    st.download_button("Download Change Summary (CSV)", csv_data, file_name="change_summary.csv")
else:
    st.info("Please upload both images to start the change detection process.")
