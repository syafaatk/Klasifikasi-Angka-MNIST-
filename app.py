import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
import cv2
import numpy as np

# Konfigurasi Halaman
st.set_page_config(page_title="Digit Recognizer", layout="centered")

# --- 1. LOAD MODEL ---
@st.cache_resource
def load_my_model():
    # Pastikan file model_mnist.h5 ada di folder yang sama
    try:
        return tf.keras.models.load_model('model_mnist.h5')
    except Exception as e:
        st.error(f"Gagal memuat model: {e}. Pastikan file 'model_mnist.h5' sudah diunggah.")
        return None

model = load_my_model()

# --- 2. ANTARMUKA PENGGUNA ---
st.title("🔢 MNIST Digit Recognizer")
st.write("Gambar sebuah angka (0-9) di dalam kotak hitam di bawah ini, lalu klik **Predict**.")

# Membuat layout kolom
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Kanvas Gambar")
    # Komponen Kanvas Streamlit
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
    st.subheader("Hasil Prediksi")
    if st.button("Predict"):
        if canvas_result.image_data is not None and model is not None:
            # --- 3. PREPROCESSING ---
            # Ambil data gambar (RGBA)
            img = canvas_result.image_data.astype(np.uint8)
            
            # Konversi ke Grayscale
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
            
            # Resize ke 28x28 sesuai standar MNIST
            img_resized = cv2.resize(img_gray, (28, 28), interpolation=cv2.INTER_AREA)
            
            # Normalisasi
            img_final = img_resized / 255.0
            
            # Reshape untuk input model (Batch, Height, Width)
            img_input = img_final.reshape(1, 28, 28)
            
            # --- 4. PREDIKSI ---
            prediction = model.predict(img_input)
            predicted_digit = np.argmax(prediction)
            confidence = np.max(prediction) * 100
            
            # --- 5. TAMPILKAN HASIL ---
            st.markdown(f"## Prediksi: **{predicted_digit}**")
            st.write(f"Tingkat Keyakinan: {confidence:.1f}%")
            
            # Progress bar untuk visualisasi keyakinan
            st.progress(int(confidence))
            
            # Menampilkan gambar kecil yang diproses AI
            st.image(img_resized, caption="Input (28x28)", width=100)
        else:
            st.warning("Silakan gambar angka terlebih dahulu!")

# Catatan kaki
st.info("Catatan: Model ini dilatih menggunakan dataset MNIST standar.")