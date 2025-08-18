# app.py - Simple Version
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json
import os
import io

# ---------------- CONFIG ----------------
MODEL_PATH = "saved_model/mobilenet_articleType"
LABELS_PATH = "saved_model/articleType_labels.json"
IMG_SIZE = (160, 160)

# ---------------- LOAD MODEL & LABELS ----------------
@st.cache_resource
def load_artifacts():
    try:
        # Load model
        model = tf.keras.models.load_model(MODEL_PATH)
        
        # Load labels
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            idx_to_class = json.load(f)
        idx_to_class = {int(k): v for k,v in idx_to_class.items()}
        
        return model, idx_to_class
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.error("Please ensure model and labels files exist in saved_model/ folder")
        st.stop()

def preprocess_pil(pil_img):
    """Preprocess PIL image for MobileNetV2"""
    pil_img = pil_img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(pil_img).astype("float32")
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    return arr

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="Fashion Classifier", page_icon="👚")
st.title("👚 Fashion Image Classifier")
st.caption("Upload a clothing image and get predicted articleType.")

# Load model and labels
model, idx_to_class = load_artifacts()

# File uploader
uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded:
    # Display image
    pil_img = Image.open(uploaded)
    st.image(pil_img, caption="Uploaded Image", use_container_width=True)
    
    # Predict
    with st.spinner("Predicting..."):
        x = preprocess_pil(pil_img)
        preds = model.predict(x, verbose=0)[0]
        
        # Get top 3 predictions
        top3_idx = preds.argsort()[-3:][::-1]
        
        st.subheader("Predictions")
        for idx in top3_idx:
            label = idx_to_class[idx]
            confidence = float(preds[idx])
            st.write(f"**{label}**: {confidence*100:.1f}%")
            st.progress(confidence)
