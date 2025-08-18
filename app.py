# app.py - Keras 3 Compatible
import streamlit as st
import tensorflow as tf
import keras
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
        # Method 1: Try using TFSMLayer for Keras 3
        try:
            st.info("Loading model with TFSMLayer (Keras 3)...")
            model_layer = keras.layers.TFSMLayer(
                MODEL_PATH, 
                call_endpoint='serving_default'
            )
            
            # Create a simple wrapper model
            inputs = keras.Input(shape=(*IMG_SIZE, 3))
            # TFSMLayer expects specific input format
            outputs = model_layer(inputs)
            # Extract the prediction tensor (might need adjustment based on your model)
            if isinstance(outputs, dict):
                # If output is dict, get the prediction tensor
                prediction_key = list(outputs.keys())[0]
                outputs = outputs[prediction_key]
            
            model = keras.Model(inputs=inputs, outputs=outputs)
            st.success("✅ Model loaded with TFSMLayer!")
            
        except Exception as e1:
            st.warning(f"TFSMLayer failed: {e1}")
            
            # Method 2: Try tf.saved_model.load
            try:
                st.info("Loading model with tf.saved_model.load...")
                loaded_model = tf.saved_model.load(MODEL_PATH)
                
                # Create wrapper function
                @tf.function
                def predict_fn(x):
                    return loaded_model(x)
                
                # Create a custom model class
                class SavedModelWrapper:
                    def __init__(self, loaded_model):
                        self.model = loaded_model
                    
                    def predict(self, x, **kwargs):
                        # Convert to tensor if needed
                        if isinstance(x, np.ndarray):
                            x = tf.constant(x)
                        
                        # Get prediction
                        try:
                            # Try direct call
                            result = self.model(x)
                        except:
                            # Try with signatures
                            infer = self.model.signatures["serving_default"]
                            result = infer(tf.constant(x))
                            # Extract tensor from dict if needed
                            if isinstance(result, dict):
                                result = list(result.values())[0]
                        
                        return result.numpy()
                
                model = SavedModelWrapper(loaded_model)
                st.success("✅ Model loaded with tf.saved_model.load!")
                
            except Exception as e2:
                st.error(f"tf.saved_model.load failed: {e2}")
                
                # Method 3: Force load with older TF method
                try:
                    st.info("Trying compatibility mode...")
                    # Disable Keras 3 temporarily
                    os.environ['TF_USE_LEGACY_KERAS'] = '1'
                    model = tf.keras.models.load_model(MODEL_PATH)
                    st.success("✅ Model loaded in compatibility mode!")
                except Exception as e3:
                    st.error(f"All loading methods failed: {e3}")
                    st.stop()
        
        # Load labels
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            idx_to_class = json.load(f)
        idx_to_class = {int(k): v for k,v in idx_to_class.items()}
        st.success("✅ Labels loaded successfully!")
        
        return model, idx_to_class
        
    except Exception as e:
        st.error(f"Error loading artifacts: {str(e)}")
        st.error("Consider converting your model to .keras or .h5 format")
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

# Version info
st.sidebar.write(f"TensorFlow: {tf.__version__}")
st.sidebar.write(f"Keras: {keras.__version__}")

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
        try:
            x = preprocess_pil(pil_img)
            preds = model.predict(x, verbose=0)
            
            # Handle different prediction formats
            if len(preds.shape) > 1:
                preds = preds[0]
            
            # Get top 3 predictions
            top3_idx = preds.argsort()[-3:][::-1]
            
            st.subheader("Predictions")
            for idx in top3_idx:
                label = idx_to_class[idx]
                confidence = float(preds[idx])
                st.write(f"**{label}**: {confidence*100:.1f}%")
                st.progress(confidence)
                
        except Exception as pred_error:
            st.error(f"Prediction failed: {str(pred_error)}")
            st.error("Model prediction format might need adjustment")
