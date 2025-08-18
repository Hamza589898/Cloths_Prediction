# app.py
import streamlit as st, numpy as np, json, tensorflow as tf
from PIL import Image
import io, os

# ================== CONFIG ==================
OUT_DIR     = r"E:\hamza\freelancing_project\CNN_cloathing_project\saved_model"
MODEL_PATH  = os.path.join(OUT_DIR, "mobilenet_articleType.h5")
LABELS_PATH = os.path.join(OUT_DIR, "articleType_labels.json")
IMG_SIZE    = (160,160)

# ================== LOAD MODEL & LABELS ==================
@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found at: {MODEL_PATH}")
        st.stop()
    if not os.path.exists(LABELS_PATH):
        st.error(f"❌ Labels file not found at: {LABELS_PATH}")
        st.stop()

    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        idx_to_class = json.load(f)
    idx_to_class = {int(k): v for k, v in idx_to_class.items()}
    return model, idx_to_class

def preprocess_pil(pil_img):
    pil_img = pil_img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(pil_img).astype("float32")
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    return arr

# ================== STREAMLIT UI ==================
st.set_page_config(page_title="Fashion Classifier", page_icon="👚", layout="centered")
st.title("👚 Fashion Image Classifier")
st.caption("Upload a clothing product image. The model predicts the most likely **articleType**.")

# Load once
model, idx_to_class = load_artifacts()

uploaded = st.file_uploader("📤 Upload an image (JPG/PNG)", type=["jpg","jpeg","png"])

if uploaded:
    # Show uploaded image
    img_bytes = uploaded.read()
    pil_img = Image.open(io.BytesIO(img_bytes))
    st.image(pil_img, caption="Uploaded Image", use_container_width=True)

    # Predict
    x = preprocess_pil(pil_img)
    preds = model.predict(x, verbose=0)[0]  # shape: (num_classes,)
    top3_idx = preds.argsort()[-3:][::-1]
    top_rows = [(idx_to_class[i], float(preds[i])) for i in top3_idx]

    st.subheader("🔮 Predictions")
    for label, p in top_rows:
        st.write(f"**{label}** — {p*100:.1f}%")
        st.progress(float(p))
