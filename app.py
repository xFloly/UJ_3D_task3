import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

st.set_page_config(page_title="Image Analyzer", layout="wide")

st.title("Analiza histogramu i poprawa ostrości obrazu")

uploaded_file = st.file_uploader("Wczytaj obraz:", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    #Wczytanie
    image = Image.open(uploaded_file)
    image_np = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    st.image(image_np, caption="Oryginalny obraz", width=250)

    # Analiza histogramu
    st.subheader("Histogram obrazu (in grayscale)")

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(hist, color='black')
    ax.set_title("Histogram jasności")
    ax.set_xlabel("Poziom jasności (0-255)")
    ax.set_ylabel("Liczba pikseli")
    col1, col2 = st.columns([1, 4]) 
    with col1:
        st.pyplot(fig)  


    # Poprawa ostrości
    st.subheader("Poprawa ostrości")

    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(image_np, -1, kernel)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image_np, caption="Przed wyostrzeniem", use_container_width=True)
    with col2:
        st.image(sharpened, caption="Po wyostrzeniu", use_container_width=True)
