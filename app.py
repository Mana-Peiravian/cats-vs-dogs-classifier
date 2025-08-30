import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.title("🐱🐶 Cat vs Dog Classifier")

model = tf.keras.models.load_model("models/cats_dogs_resnet.h5")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).resize((150, 150))
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(image)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    label = "Dog 🐶" if prediction > 0.5 else "Cat 🐱"
    st.subheader(f"Prediction: {label}")
