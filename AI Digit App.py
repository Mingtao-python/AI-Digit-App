import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
import cv2, time

model = tf.keras.models.load_model("cnn_mnist.keras")

st.title("Welcome to use my MNIST Digit Recognizer!")
st.subheader("MNIST Handwritten Digit Recognition")
st.write("Draw a digit (0-9) on the canvas below.")

canvas_result = st_canvas(fill_color="rgba(0, 0, 0, 0)", stroke_width=12, stroke_color="white", background_color="black", width=280, height=280, drawing_mode="freedraw", key="canvas",)

def preprocess(img):
    img = img.convert("L")
    img_np = np.array(img)
    _, thresh = cv2.threshold(img_np, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    digit = img_np[y:y+h, x:x+w]
    digit = Image.fromarray(digit)
    digit = ImageOps.expand(digit, border=20, fill=0)
    digit = digit.resize((28, 28))
    return digit

if canvas_result.image_data is not None:
    raw_img = Image.fromarray(canvas_result.image_data.astype("uint8"))
    processed = preprocess(raw_img)

    if processed is not None:
        st.image(processed, caption="Processed Input Image", width=150)
        img_array = np.array(processed).reshape(1, 28, 28, 1).astype("float32") / 255.0
        time.sleep(1)
        pred = np.argmax(model.predict(img_array), axis=1)[0]
        st.subheader(f"Prediction: {pred}")
