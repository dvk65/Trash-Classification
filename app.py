import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications import ResNet50
from PIL import Image
import numpy as np
import os


resize_and_rescale = tf.keras.Sequential([
    tf.keras.layers.Resizing(224, 224),
    tf.keras.layers.Rescaling(preprocess_input),
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomContrast(0.2),
    tf.keras.layers.RandomBrightness(0.2)
])

pretrained_model = ResNet50(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet',
    pooling='avg'
)

inputs = pretrained_model.input
x = resize_and_rescale(inputs)

x = Dense(256, activation='relu')(pretrained_model.output)
x = Dropout(0.2)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.2)(x)

outputs = Dense(4, activation='softmax')(x) #4 targets so far

model = Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer=Adam(0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.load_weights("trashclassify_weights.weights.h5")

st.write("Model Loaded Successfully!")

st.write("# Trash Classification App")

# @st.cache_resource
# def load_model():
#     return tf.keras.models.load_weights("trashclassify.keras")

# model = load_model()

# --- Camera input ---
camera_photo = st.camera_input("Take a photo")

# --- File uploader ---
uploaded_photo = st.file_uploader("Or upload an image", type=["jpg", "jpeg", "png"])

# --- Use whichever input is available ---
if camera_photo is not None:
    uploaded_file = camera_photo
elif uploaded_photo is not None:
    uploaded_file = uploaded_photo
else:
    uploaded_file = None

# --- Display image if available ---
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Your Image", use_column_width=True)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = image.resize((224, 224))  # adjust to your model's input size
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    pred_index = np.argmax(prediction[0]) # get prediction index of highest probablity
    class_names = ["bottles", "cans", "cardboard", "cups"]
    label = class_names[pred_index]

    st.write("Predicted Label: ", label)
    st.write("Prediction Probabilities: ", prediction)

    st.markdown(f"### 🏷 Prediction: **{label}**")
