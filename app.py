import streamlit as st
import tensorflow as tf
# from PIL import Image
# import numpy as np
import os

st.write("# Trash Classification App")
	
# 
# pillow
# numpy
# keras
# huggingface-hub
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("trashclassify.keras")

model = load_model()

# --- Camera input ---
camera_photo = st.camera_input("Take a photo")

# --- File uploader ---
uploaded_photo = st.file_uploader("Or upload an image", type=["jpg", "jpeg", "png"])

# # --- Use whichever input is available ---
# if camera_photo is not None:
#     uploaded_file = camera_photo
# elif uploaded_photo is not None:
#     uploaded_file = uploaded_photo
# else:
#     uploaded_file = None

# # --- Display image if available ---
# if uploaded_file:
#     image = Image.open(uploaded_file).convert("RGB")
#     st.image(image, caption="Your Image", use_column_width=True)

# if uploaded_file is not None:
#     image = Image.open(uploaded_file).convert("RGB")
#     st.image(image, caption="Uploaded Image", use_column_width=True)

#     # Preprocess
#     img = image.resize((224, 224))  # adjust to your model's input size
#     img_array = np.array(img) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)

#     # Prediction
#     prediction = model.predict(img_array)
#     label = "Recycle" if prediction[0][0] > 0.5 else "General Trash"  # adjust based on your model’s output shape

#     st.markdown(f"### 🏷 Prediction: **{label}**")
