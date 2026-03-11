import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

model = load_model("mnist_model.h5")

st.title("MNIST Digit Recognition")

canvas = st_canvas(
    fill_color="black",
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas"
)

if canvas.image_data is not None:

    # convert canvas to image
    img = Image.fromarray(canvas.image_data.astype('uint8'))

    # convert to grayscale
    img = img.convert("L")

    # resize to MNIST size
    img = img.resize((28,28))

    # convert to numpy
    img = np.array(img)

    # invert colors
    img = 255 - img

    # normalize
    img = img / 255.0

    # reshape for CNN
    img = img.reshape(1,28,28,1)

    if st.button("Predict"):

        prediction = model.predict(img)
        digit = np.argmax(prediction)

        st.success(f"Predicted Digit: {digit}")
