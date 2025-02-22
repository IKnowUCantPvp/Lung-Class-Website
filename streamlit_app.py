import streamlit as st
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
import os
import tempfile
import matplotlib.pyplot as plt
import pandas as pd
from sympy.abc import epsilon

st.title('🩺 Lung Sound Classification App')
st.write(
    "This app is built off of a research project exploring the usage of Per Channel Energy Normalization "
    "to classify lung sounds. Upload an audio file below to classify it!"
)

# Load your pre-trained model from the repository.
# The model directory should contain the saved_model.pb file and related assets.
MODEL_DIR = "Conv2DOldPCEN"
model = tf.saved_model.load(MODEL_DIR)
# Get the default serving function from the model.
inference_func = model.signatures["serving_default"]

# Mapping from class indices to human-readable labels.
CLASS_NAMES = {
    0: "Asthma",
    1: "Bronchiectasis",
    2: "Bronchiolitis",
    3: "COPD",
    4: "Healthy",
    5: "Heart Failure",
    6: "Lung Fibrosis",
    7: "Pleural Effusion",
    8: "Pneumonia",
    9: "URTI",
}


def load_audio_waveform(file_path):
    """
    Loads an audio file, resamples it to 8 kHz, and adjusts its duration to 6 seconds.
    Returns the raw waveform (1D numpy array) and the sampling rate.
    """
    target_sr = 8000  # 8 kHz sample rate.
    target_duration = 6  # 6 seconds duration.
    target_length = target_sr * target_duration  # 48000 samples.

    y, sr = librosa.load(file_path, sr=target_sr)
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)), mode='constant')
    else:
        y = y[:target_length]
    return y, sr


def preprocess_audio(file_path):
    """
    Loads the audio and returns a tensor of shape (1, 48000, 1)
    which matches the model's expected input.
    """
    y, _ = load_audio_waveform(file_path)
    # Add channel dimension to get shape (48000, 1)
    y = np.expand_dims(y, axis=-1)
    # Add batch dimension: final shape (1, 48000, 1)
    return y[np.newaxis, ...]


def predict_class(audio_file):
    """
    Writes the uploaded audio file to a temporary file, preprocesses it,
    and uses the loaded model to predict the lung sound classification.
    Also returns the raw waveform and sampling rate for further visualization.
    """
    # Save the uploaded audio file temporarily.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_file.read())
        tmp_path = tmp.name

    # Load audio for visualization.
    waveform, sr = load_audio_waveform(tmp_path)
    # Preprocess for prediction.
    input_data = preprocess_audio(tmp_path)
    os.remove(tmp_path)

    # Convert the waveform to a TensorFlow tensor.
    input_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)

    # Run inference using the model's serving function.
    prediction = inference_func(input_tensor)
    # Extract the output tensor (assumes a single output tensor is returned).
    pred_tensor = list(prediction.values())[0]
    pred_probs = pred_tensor.numpy()[0]
    predicted_index = np.argmax(pred_probs)
    predicted_label = CLASS_NAMES.get(predicted_index, "Unknown")
    return predicted_label, pred_probs, waveform, sr


# Streamlit UI for file upload and prediction.
st.write("Upload an audio file to classify the lung sound:")
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')

    if st.button("Predict"):
        with st.spinner("Processing..."):
            predicted_label, pred_probs, waveform, sr = predict_class(uploaded_file)

        # Calculate confidence and some additional metrics.
        confidence = np.max(pred_probs) * 100
        duration = len(waveform) / sr
        rms = np.sqrt(np.mean(waveform ** 2))

        # Display prediction results.
        st.success(f"Predicted lung sound classification: **{predicted_label}**")
        st.write(f"Model Confidence: **{confidence:.2f}%**")
        st.write(f"**Audio Duration:** {duration:.2f} seconds")
        st.write(f"**Sampling Rate:** {sr} Hz")
        st.write(f"**RMS Amplitude:** {rms:.4f}")

        # Display probability distribution as a bar chart.
        df = pd.DataFrame({
            "Probability": pred_probs
        }, index=list(CLASS_NAMES.values()))
        st.write("### Probability Distribution for All Classes")
        st.bar_chart(df)

        # Plot the audio waveform.
        fig_wave, ax_wave = plt.subplots(figsize=(10, 4))
        times = np.linspace(0, duration, num=len(waveform))
        ax_wave.plot(times, waveform)
        ax_wave.set_title("Audio Waveform")
        ax_wave.set_xlabel("Time (s)")
        ax_wave.set_ylabel("Amplitude")
        st.pyplot(fig_wave)

        # Compute a log mel spectrogram.
        n_fft = 2048
        hop_length = 512
        n_mels = 128
        mel_spec = librosa.feature.melspectrogram(y=waveform, sr=sr, n_fft=n_fft,
                                                  hop_length=hop_length, n_mels=n_mels)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        # Compute a PCEN spectrogram from the same mel spectrogram.
        pcen_spec = librosa.core.pcen(mel_spec, time_constant=0.007, power=0.85, bias=5.5, gain=0.7, eps=1E-08)

        # Plot both spectrograms side by side.
        fig_spec, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        img1 = librosa.display.specshow(log_mel_spec, sr=sr, x_axis='time', y_axis='mel', ax=ax1)
        ax1.set_title("Log Mel Spectrogram")
        fig_spec.colorbar(img1, ax=ax1, format="%+2.0f dB")

        img2 = librosa.display.specshow(pcen_spec, sr=sr, x_axis='time', y_axis='mel', ax=ax2)
        ax2.set_title("PCEN Spectrogram")
        fig_spec.colorbar(img2, ax=ax2, format="%+2.0f")

        st.pyplot(fig_spec)
