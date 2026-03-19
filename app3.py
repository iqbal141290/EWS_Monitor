import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import time

st.set_page_config(page_title="EWS Time-Series V3", layout="wide")
st.title("🚢 Real-Time EWS Monitor (LSTM)")

# 1. Load Model & Scaler
@st.cache_resource
def load_ai():
    model = load_model('model_lstm_v3.h5')
    scaler = joblib.load('scaler_v3.sav')
    return model, scaler

model, scaler = load_ai()

# 2. Simulasi Data Time Series (Nanti ganti dengan data AQWA/Sensor)
# Kita buat data sinus dummy untuk simulasi di web
t = np.linspace(0, 100, 1000)
data_dummy = (np.sin(t) + 0.5 * np.sin(2*t)).reshape(-1, 1)

# 3. Sidebar Control
st.sidebar.header("Kontrol Monitor")
run_monitor = st.sidebar.checkbox("Mulai Monitoring Real-Time", value=False)
speed_sim = st.sidebar.slider("Kecepatan Update (detik)", 0.1, 1.0, 0.5)

# 4. Dashboard Main
placeholder = st.empty() # Wadah untuk grafik yang berganti-ganti

if run_monitor:
    # Loop Animasi
    for i in range(100, 200): # Simulasi berjalan 100 detik
        win = 20
        # Ambil data masa lalu
        history = data_dummy[i : i+win]
        
        # Prediksi AI
        scaled_in = scaler.transform(history).reshape(1, win, 1)
        pred_scaled = model.predict(scaled_in, verbose=0)
        y_pred = scaler.inverse_transform(pred_scaled).flatten()
        
        # Buat Plot
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(range(i, i+win), history, 'b-o', label="Data Sensor")
        ax.plot(i+win, y_pred, 'rx', markersize=12, markeredgewidth=3, label="Prediksi AI (Next)")
        
        ax.set_ylim(-2, 2)
        ax.set_xlim(i-5, i+win+5)
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_title(f"Monitoring Detik ke-{i}")
        
        # Tampilkan ke Streamlit
        with placeholder.container():
            st.pyplot(fig)
            if y_pred > 1.2: # Batas Aman
                st.error(f"🚨 PERINGATAN: Prediksi Oleng Bahaya ({y_pred[0]:.2f}°)")
            else:
                st.success("✅ Kondisi Stabil")
        
        plt.close(fig)
        time.sleep(speed_sim) # Memberi jeda agar terlihat seperti animasi
else:
    st.info("Klik 'Mulai Monitoring' di sidebar untuk melihat animasi prediksi.")
