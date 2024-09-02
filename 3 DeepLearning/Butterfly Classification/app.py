import streamlit as st
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import tensorflow as tf

# Load the model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('model_butter.h5')  # Path to your model
    return model

model = load_model()

# Class labels
class_labels = [
    'SOUTHERN DOGFACE', 'ADONIS', 'BROWN SIPROETA', 'MONARCH', 'GREEN CELLED CATTLEHEART', 
    'CAIRNS BIRDWING', 'EASTERN DAPPLE WHITE', 'RED POSTMAN', 'MANGROVE SKIPPER', 
    'BLACK HAIRSTREAK', 'CABBAGE WHITE', 'RED ADMIRAL', 'PAINTED LADY', 'PAPER KITE', 
    'SOOTYWING', 'PINE WHITE', 'PEACOCK', 'CHECQUERED SKIPPER', 'JULIA', 
    'COMMON WOOD-NYMPH', 'BLUE MORPHO', 'CLOUDED SULPHUR', 'STRAITED QUEEN', 
    'ORANGE OAKLEAF', 'PURPLISH COPPER', 'ATALA', 'IPHICLUS SISTER', 'DANAID EGGFLY', 
    'LARGE MARBLE', 'PIPEVINE SWALLOW', 'BLUE SPOTTED CROW', 'RED CRACKER', 
    'QUESTION MARK', 'CRIMSON PATCH', 'BANDED PEACOCK', 'SCARCE SWALLOW', 'COPPER TAIL', 
    'GREAT JAY', 'INDRA SWALLOW', 'VICEROY', 'MALACHITE', 'APPOLLO', 
    'TWO BARRED FLASHER', 'MOURNING CLOAK', 'TROPICAL LEAFWING', 'POPINJAY', 
    'ORANGE TIP', 'GOLD BANDED', 'BECKERS WHITE', 'RED SPOTTED PURPLE', 
    'MILBERTS TORTOISESHELL', 'SILVER SPOT SKIPPER', 'AMERICAN SNOOT', 'AN 88', 
    'ULYSES', 'COMMON BANDED AWL', 'CRECENT', 'METALMARK', 'SLEEPY ORANGE', 
    'PURPLE HAIRSTREAK', 'ELBOWED PIERROT', 'GREAT EGGFLY', 'ORCHARD SWALLOW', 
    'ZEBRA LONG WING', 'WOOD SATYR', 'MESTRA', 'EASTERN PINE ELFIN', 
    'EASTERN COMA', 'YELLOW SWALLOW TAIL', 'CLEOPATRA', 'GREY HAIRSTREAK', 
    'BANDED ORANGE HELICONIAN', 'AFRICAN GIANT SWALLOWTAIL', 'CHESTNUT', 
    'CLODIUS PARNASSIAN'
]

# Title
st.title("🦋 Butterfly Classifier")

# Image Upload
st.header("Upload an Image")
uploaded_file = st.file_uploader("Please upload a butterfly image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    st.write("🔍 **Analyzing the image...**")

    # Preprocess the image
    img = img.resize((128, 128))  # Resize according to how your model was trained
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Predict
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]

    # Display the result
    st.write(f"🦋 **It's a {class_labels[predicted_class]}!**")
else:
    st.write("👈 Upload an image to get started!")
