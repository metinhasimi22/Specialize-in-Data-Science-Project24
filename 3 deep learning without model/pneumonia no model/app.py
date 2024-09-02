import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image  # Pillow'u import edin

# Modeli yükle
model = load_model('model.h5')  # Modelinizi burada yükleyin

# Uygulama başlığı
st.title("🩺 Pneumonia Classification App")
st.write("📷 **X-ray görüntüsünü yükleyin ve pnömoni olup olmadığını kontrol edin.**")

# Dosya yükleme alanı
uploaded_file = st.file_uploader("🖼️ Resim yükle", type=["jpeg", "jpg", "png"])

if uploaded_file is not None:
    # Resmi yükle ve Pillow kullanarak aç
    img = Image.open(uploaded_file)  
    img = img.resize((150, 150))  # Resmi hedef boyuta yeniden boyutlandırın
    st.image(img, caption='Yüklenen Resim', use_column_width=True)
    st.write("")
    st.write("🔍 **Tahmin ediliyor...**")

    # Resmi işleyin ve model ile tahmin yapın
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalizasyon

    prediction = model.predict(img_array)
    class_names = ['🟢 Normal', '🔴 Pneumonia']
    predicted_class = class_names[int(prediction[0] > 0.5)]

    st.success(f"✅ **Tahmin: {predicted_class}**" if predicted_class == '🟢 Normal' else f"🚨 **Tahmin: {predicted_class}**")