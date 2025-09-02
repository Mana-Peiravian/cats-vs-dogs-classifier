import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# --- Page Config ---
st.set_page_config(page_title="🐱🐶 Cat vs Dog Classifier", layout="centered")

# --- Load Model ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/cats_dogs_resnet.h5")

model = load_model()

# --- Custom CSS ---
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}
.container {
    background: rgba(255, 255, 255, 0.95);
    padding: 20px;
    border-radius: 20px;
    max-width: 800px;
    margin: 40px auto;
}
h1 {
    text-align: center;
    font-size: 2.5em;
    background: linear-gradient(45deg, #667eea, #764ba2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.upload-area {
    border: 3px dashed #667eea;
    border-radius: 15px;
    padding: 40px;
    text-align: center;
    margin: 20px 0;
    background: rgba(102, 126, 234, 0.05);
}
.result-cat { background: linear-gradient(45deg, #ff9a9e, #fad0c4); border-left: 5px solid #ff6b6b; padding:15px; border-radius:10px; text-align:center;}
.result-dog { background: linear-gradient(45deg, #a8edea, #fed6e3); border-left: 5px solid #4ecdc4; padding:15px; border-radius:10px; text-align:center;}
.loading { text-align: center; padding: 20px; }
.spinner { border: 4px solid #f3f3f3; border-top: 4px solid #667eea; border-radius: 50%; width: 50px; height: 50px; animation: spin 1s linear infinite; margin: 0 auto; }
@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
</style>
""", unsafe_allow_html=True)

# --- Container Start ---
st.markdown('<div class="container">', unsafe_allow_html=True)
st.markdown("<h1>🐱🐶 Cat vs Dog Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Upload an image and AI will tell you if it's a cat or dog!</p>", unsafe_allow_html=True)

# --- File uploader ---
uploaded_file = st.file_uploader("📸 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # --- Show loading spinner ---
    with st.spinner("Analyzing image with AI..."):
        image = Image.open(uploaded_file).convert("RGB").resize((150, 150))
        st.image(image, caption="Uploaded Image", use_column_width=True)

        img_array = np.array(image)/255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)[0][0]

        if prediction > 0.5:
            label = f"🐶 This is a DOG! Confidence: {prediction*100:.1f}%"
            st.markdown(f'<div class="result-dog">{label}</div>', unsafe_allow_html=True)
        else:
            label = f"🐱 This is a CAT! Confidence: {(1-prediction)*100:.1f}%"
            st.markdown(f'<div class="result-cat">{label}</div>', unsafe_allow_html=True)

# --- Container End ---
st.markdown('</div>', unsafe_allow_html=True)
