import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import time

# 1. Konfigurasi Halaman Dasar
st.set_page_config(page_title="EWS Time-Series V3", layout="wide")
st.title("🚢 Real-Time EWS Monitor (LSTM)")
st.markdown("Sistem Prediksi Gerakan Kapal Berbasis Deret Waktu (*Time-Series*) menggunakan LSTM.")

# 2. Fungsi Load Model & Scaler (Memakai @st.cache_resource agar hanya load sekali saja)
@st.cache_resource
def load_ai():
    try:
        # Gunakan format .keras sesuai standar terbaru agar tidak error di Streamlit
        model = load_model('model_lstm_v3.keras') 
        scaler = joblib.load('scaler_v3.sav')
        return model, scaler
    except Exception as e:
        st.error(f"Gagal memuat file model AI: {e}")
        st.info("Pastikan file 'model_lstm_v3.keras' dan 'scaler_v3.sav' sudah ada di folder GitHub yang sama.")
        return None, None

model, scaler = load_ai()

# 3. Kontrol Sidebar
st.sidebar.header("🕹️ Kontrol Dashboard")
run_monitor = st.sidebar.checkbox("Mulai Monitoring Real-Time", value=False)
speed_sim = st.sidebar.slider("Kecepatan Update (Detik)", 0.1, 1.0, 0.5)

# 4. Wadah Grafik (Placeholder) agar grafik tertimpa secara mulus (animasi)
placeholder = st.empty()

# 5. Simulasi Data (Gunakan Logika Sinus yang Sama dengan di Colab untuk Tes Awal)
# Dalam kenyataannya, nanti ini diganti dengan data sensor kapal asli
t = np.linspace(0, 100, 1000)
data_dummy = (np.sin(t) + 0.5 * np.sin(2*t)).reshape(-1, 1)

if run_monitor and model is not None:
    # Loop untuk membuat grafik bergerak (Animasi)
    for i in range(100, 300):
        if not run_monitor: break # Berhenti jika checkbox dimatikan
        
        win = 20 # Sesuai dengan jendela waktu saat training
        # Ambil potongan data masa lalu (20 detik)
        history = data_dummy[i : i+win]
        
        # Proses Prediksi Masa Depan (Menebak 1 langkah depan)
        scaled_in = scaler.transform(history).reshape(1, win, 1)
        pred_scaled = model.predict(scaled_in, verbose=0)
        y_pred = scaler.inverse_transform(pred_scaled).flatten()[0]
        
        # PROSES VISUALISASI (Matplotlib)
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Garis Biru: 20 detik sejarah
        time_axis = range(i, i+win)
        ax.plot(time_axis, history, 'b-o', label="Data Sensor (History)")
        
        # Titik Merah: Tebakan AI untuk detik ke-21 (Masa Depan)
        ax.plot(i+win, y_pred, 'rx', markersize=12, markeredgewidth=3, label="Prediksi AI (Next)")
        
        # Styling Grafik
        ax.set_ylim(-2, 2) # Rentang gerakan roll/pitch kapal
        ax.set_xlim(i-5, i+win+5) # Sumbu X bergerak dinamis
        ax.set_xlabel("Waktu (Detik)")
        ax.set_ylabel("Respon (Derajat)")
        ax.legend(loc='upper left')
        ax.grid(alpha=0.3)
        ax.set_title(f"Monitoring Real-Time | Detik ke-{i}")
        
        # Tampilkan ke Streamlit di dalam placeholder
        with placeholder.container():
            st.pyplot(fig)
            
            # --- LOGIKA EARLY WARNING ---
            if y_pred > 1.2: # Misal batas miring aman adalah 1.2 derajat
                st.error(f"🚨 BAHAYA: Prediksi Oleng Berlebih ({y_pred:.2f}°)")
                st.warning("Rekomendasi: Segera kurangi kecepatan atau ubah haluan!")
            else:
                st.success(f"✅ Kondisi Aman. Prediksi Berikutnya: {y_pred:.2f}°")
        
        plt.close(fig) # Penting: Tutup plot agar memori tidak penuh
        time.sleep(speed_sim) # Memberi jeda waktu simulasi

elif model is None:
    st.warning("Sistem belum siap. Pastikan model AI sudah terunggah di GitHub.")
else:
    st.info("Silakan klik 'Mulai Monitoring Real-Time' pada sidebar untuk memulai simulasi.")

# Footer
st.divider()
st.caption("Dashboard Seakeeping V3 - Dikembangkan untuk Analisis Kapal Ikan Kecil")
