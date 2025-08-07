import streamlit as st
import mne
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.signal import stft
import matplotlib.pyplot as plt
import pickle
import os
import tempfile

st.set_page_config(layout="wide")
st.title("🧠 Klasifikasi Sinyal EEG (.edf)")

# -------------------------
# Fungsi: Bandpass Filter
# -------------------------
def bandpass_filter(raw, l_freq=1.0, h_freq=40.0):
    raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin', verbose=False)
    return raw

# -------------------------
# Fungsi: Ekstraksi Fitur
# -------------------------
def extract_features(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose=False)
    raw = bandpass_filter(raw, 1.0, 40.0)
    data = raw.get_data()
    sfreq = raw.info['sfreq']

    features = []
    nperseg = int(sfreq * 2)
    noverlap = int(nperseg * 0.5)

    for channel_data in data:
        mean_val = np.mean(channel_data)
        std_val = np.std(channel_data)
        skew_val = skew(channel_data)
        kurt_val = kurtosis(channel_data)

        f, t, Zxx = stft(channel_data, fs=sfreq, nperseg=nperseg, noverlap=noverlap)
        power_spectrum = np.abs(Zxx) ** 2
        avg_power = np.mean(power_spectrum, axis=1)
        dominant_freq = float(f[np.argmax(avg_power)])

        features.extend([mean_val, std_val, skew_val, kurt_val, dominant_freq])

    return np.array(features).reshape(1, -1), raw, data

# -------------------------
# Sidebar: Pilihan Model
# -------------------------
st.sidebar.title("🔧 Pengaturan Klasifikasi")
model_type = st.sidebar.selectbox("Pilih Metode Klasifikasi", ["SVM Linear", "SVM Polynomial", "Naive Bayes"])

# Pilihan C untuk SVM
c_value = None
if model_type in ["SVM Linear", "SVM Polynomial"]:
    c_value = st.sidebar.selectbox("Pilih Nilai Parameter C", [0.01, 0.1, 1, 10])

# -------------------------
# Load Model Sesuai Pilihan
# -------------------------
model = None
if model_type == "SVM Linear":
    model_filename = f"svm_linear_c{c_value}.pkl"
elif model_type == "SVM Polynomial":
    model_filename = f"svm_poly_c{c_value}.pkl"
else:
    model_filename = "model_nb.pkl"

if os.path.exists(model_filename):
    with open(model_filename, "rb") as f:
        model = pickle.load(f)
else:
    st.warning(f"⚠️ Model '{model_filename}' tidak ditemukan.")

# -------------------------
# Upload File EDF
# -------------------------
uploaded_file = st.file_uploader("📁 Unggah file EEG (.edf)", type=["edf"])

if uploaded_file is not None:
    st.subheader("📄 Informasi File")
    st.write(f"Nama file: **{uploaded_file.name}**")

    # Ekstraksi fitur
    features, raw, data = extract_features(uploaded_file)

    st.subheader("📦 Data Fitur untuk Prediksi")
    st.write("Shape fitur (baris = 1 sampel, kolom = fitur):", features.shape)
    st.dataframe(pd.DataFrame(features), use_container_width=True)

    st.write(f"Jumlah channel EEG: **{data.shape[0]}**")
    st.write(f"Total fitur yang dihasilkan: **{features.shape[1]}**")
    # Tabel Fitur per Channel
    st.subheader("📈 Nilai Fitur per Channel")
    channel_names = raw.ch_names
    columns = ['Mean', 'Std', 'Skewness', 'Kurtosis', 'Dominant Freq']
    feature_labels = [f"{ch}_{col}" for ch in channel_names for col in columns]
    df_features = pd.DataFrame(features, columns=feature_labels)
    st.dataframe(df_features.T, use_container_width=True)

    # Visualisasi Channel 0
    st.subheader("📊 Visualisasi Channel 0 (Setelah Bandpass Filter)")
    ch_data, times = raw[0]
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(times, ch_data[0], color='royalblue')
    ax.set_xlabel("Waktu (detik)")
    ax.set_ylabel("Amplitudo (µV)")
    ax.set_title("Sinyal EEG - Channel 0")
    st.pyplot(fig)

    # Prediksi
    if model is not None:
        prediction = model.predict(features)
        model_label = f"{model_type} (C={c_value})" if c_value is not None else model_type
        st.success(f"🧠 Prediksi Kelas dengan {model_label}: **{prediction[0]}**")
    else:
        st.error("❌ Model tidak dimuat.")
