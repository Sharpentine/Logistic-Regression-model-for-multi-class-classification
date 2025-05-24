
import streamlit as st
import numpy as np
import joblib
from PIL import Image, ImageOps

st.title("Digit Classifier (Logistic Regression)")
st.write("Upload an image of a digit (0–9)")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    model = joblib.load("logistic_digit_model.pkl")

    import cv2
    image = Image.open(uploaded_file).convert("L")  
    image = ImageOps.invert(image)  
    image = image.resize((28, 28))
    
    img_array = np.array(image).astype("uint8")
    _, img_array = cv2.threshold(img_array, 128, 255, cv2.THRESH_BINARY)
    
    img_array = np.array(image).astype("float32") / 255.0
    img_flat = img_array.reshape(1, -1)

    prediction = model.predict(img_flat)[0]
    st.image(image, caption="Processed Image", width=150)
    st.write(f"### Predicted Digit: {prediction}")
