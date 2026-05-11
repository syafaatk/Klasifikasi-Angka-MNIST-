import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
import cv2
import numpy as np

# --- 1. KONFIGURASI MODEL ---
@st.cache_resource
def load_my_model():
    # Pastikan Anda sudah menyimpan model dari Colab dengan model.save('model_mnist.h5')
    try:
        return tf.keras.models.load_model('model_mnist.h5')
    except:
        st.error("File 'model_mnist.h5' tidak ditemukan! Silakan upload file model Anda.")
        return None

model = load_my_model()

# --- 2. TAMPILAN DASHBOARD ---
st.title("🔢 Digit Recognizer Portofolio")
st.write("Gambar angka di dalam kotak hitam, lalu tekan tombol **Predict**.")

# Membuat layout kolom
col1, col2 = st.columns([1, 1])

with col1:
    # Komponen Kanvas untuk Menggambar
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=15,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

with col2:
    if st.button("Predict"):
        if canvas_result.image_data is not None and model is not None:
            # --- 3. PREPROCESSING ---
            # Ambil gambar dari kanvas (RGBA)
            img = canvas_result.image_data.astype(np.uint8)
            
            # Ubah ke Grayscale (hitam putih)
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
            
            # Resize ke 28x28 sesuai standar MNIST
            img_resized = cv2.resize(img_gray, (28, 28), interpolation=cv2.INTER_AREA)
            
            # Normalisasi 0-1
            img_normalized = img_resized / 255.0
            
            # Reshape untuk input model (Batch, Height, Width)
            img_input = img_normalized.reshape(1, 28, 28)
            
            # --- 4. PREDIKSI ---
            prediction = model.predict(img_input)
            predicted_digit = np.argmax(prediction)
            confidence = np.max(prediction) * 100
            
            # --- 5. TAMPILKAN HASIL ---
            st.success(f"### Hasil Prediksi: {predicted_digit}")
            st.write(f"Confidence: **{confidence:.1f}%**")
            
            # Menampilkan apa yang dilihat AI
            st.image(img_resized, caption="Input 28x28 (Grayscale)", width=100)
        else:
            st.warning("Silakan gambar angka terlebih dahulu!")