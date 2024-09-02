import streamlit as st
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import tensorflow as tf

# Load the model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('catdogmodel.h5')  # Path to your model
    return model

model = load_model()

# Title
st.title("🐱🐶 Cat vs. Dog Classifier")

# Image Upload
st.header("Upload an Image")
uploaded_file = st.file_uploader("Please upload a cat or dog image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    st.write("🔍 **Analyzing the image...**")

    # Preprocess the image
    img = img.resize((128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Predict
    prediction = model.predict(img_array)

    # Display the result
    if prediction < 0.5:
        st.write("🐶 **It's a Dog!**")
    else:
        st.write("🐱 **It's a Cat!**")
else:
    st.write("👈 Upload an image to get started!")
