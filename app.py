import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Coastal Change Detection POC", layout="wide")
st.title("🌊 Coastal Change Detection — Data Lake POC")
st.write("Upload two satellite images of the same coastal area (different years) to detect changes.")

img1_file = st.file_uploader("Upload First Image (Earlier Year)", type=["jpg","jpeg","png"])
img2_file = st.file_uploader("Upload Second Image (Recent Year)", type=["jpg","jpeg","png"])

if img1_file and img2_file:
    img1 = np.array(Image.open(img1_file).convert('RGB'))
    img2 = np.array(Image.open(img2_file).convert('RGB'))

    img1 = cv2.resize(img1, (800,800))
    img2 = cv2.resize(img2, (800,800))

    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    diff = cv2.absdiff(gray1, gray2)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    changed = cv2.countNonZero(thresh)
    percent_change = round((changed / thresh.size) * 100, 2)

    # --- Red Overlay via OpenCV ---
    img2_bgr = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
    red_img = np.zeros_like(img2_bgr)
    red_img[:, :, 2] = 255  # Red channel in BGR
    mask = thresh.astype(np.uint8)
    highlight_bgr = img2_bgr.copy()
    highlight_bgr[mask == 255] = red_img[mask == 255]
    highlight_rgb = cv2.cvtColor(highlight_bgr, cv2.COLOR_BGR2RGB)

    st.subheader(f"Change Detected: {percent_change}%")
    col1, col2, col3 = st.columns(3)
    with col1: st.image(img1, caption="Earlier Image", use_container_width=True)
    with col2: st.image(img2, caption="Recent Image", use_container_width=True)
    with col3: st.image(highlight_rgb, caption="Detected Changes (Red Areas)", use_container_width=True)

    csv_data = f"Location,Year1,Year2,PercentChange\nVizag,2021,2024,{percent_change}\n"
    st.download_button("Download Change Summary (CSV)", csv_data, file_name="change_summary.csv")

else:
    st.info("Please upload both images to start the change detection process.")
