import streamlit as st
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
import os
import tempfile
import matplotlib.pyplot as plt
import pandas as pd

# ------------------------------------------------------------------------------
# About This Project
# ------------------------------------------------------------------------------
st.title('🩺 Lung Sound Classification App')

# Display a banner image comparing Spectrogram vs. PCEN
st.image("Spec_PCEN_Comparison.png",
         caption="Comparison of Spectrogram and PCEN", use_container_width=True)

st.markdown("""
**About This Project**

This app is part of our research exploring advanced audio preprocessing techniques for lung sound classification, with a special focus on **Per-Channel Energy Normalization (PCEN)**. Traditional methods like log-mel spectrograms and MFCCs are often sensitive to noise, whereas PCEN adapts dynamically to suppress background noise and amplify subtle lung sounds.

**Key Highlights:**
- **Robust Feature Extraction:** PCEN enhances transient lung sounds and reduces background noise.
- **Optimized Parameters:** Fine-tuned settings (e.g., temporal smoothing *T*, dynamic compression *r*, bias, and gain) yield improved performance.
- **Data Augmentation:** Uniform 6-second segments with multiple augmentations enhance model generalization.
- **Limited Dataset – 10 Diagnostic Categories:**  
  Due to the limited data available, our model currently classifies lung sounds into **10 diagnostic categories** (Asthma, Bronchiectasis, Bronchiolitis, COPD, Healthy, Heart Failure, Lung Fibrosis, Pleural Effusion, Pneumonia, and URTI).

**Spectrogram Comparison:**  
Our study compared traditional log-mel spectrograms with PCEN spectrograms. While the log-mel spectrogram provides a view of the frequency content, it is quite sensitive to noise and amplitude variations. In contrast, the PCEN spectrogram applies dynamic normalization to each frequency channel, resulting in a normalized amplitude range that effectively minimizes background noise and highlights lung sound's subtle, transient features. This advantage is crucial in noisy clinical environments and is the reason our model leverages PCEN for input features.

For more details on our methodology and results, please refer to our [Research Paper](https://github.com/IKnowUCantPvp/Lung-Sound-Classification-PCEN/blob/main/Ma_Nathan_Paper.pdf) and view our full project on [GitHub](https://github.com/IKnowUCantPvp/Lung-Sound-Classification-PCEN.git).

Upload an audio file below to classify the lung sound based on our trained model.
""")

# ------------------------------------------------------------------------------
# Model Loading and Helper Functions
# ------------------------------------------------------------------------------

MODEL_DIR = "Conv2DOldPCEN"
model = tf.saved_model.load(MODEL_DIR)
inference_func = model.signatures["serving_default"]

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
    y = np.expand_dims(y, axis=-1)  # Shape becomes (48000, 1)
    return y[np.newaxis, ...]  # Final shape: (1, 48000, 1)


def predict_class(audio_file):
    """
    Writes the uploaded audio file to a temporary file, preprocesses it,
    and uses the loaded model to predict the lung sound classification.
    Also returns the raw waveform and sampling rate for further visualization.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_file.read())
        tmp_path = tmp.name

    # Load audio for visualization.
    waveform, sr = load_audio_waveform(tmp_path)
    # Preprocess the audio for model input.
    input_data = preprocess_audio(tmp_path)
    os.remove(tmp_path)

    input_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)
    prediction = inference_func(input_tensor)
    pred_tensor = list(prediction.values())[0]
    pred_probs = pred_tensor.numpy()[0]
    predicted_index = np.argmax(pred_probs)
    predicted_label = CLASS_NAMES.get(predicted_index, "Unknown")
    return predicted_label, pred_probs, waveform, sr


# ------------------------------------------------------------------------------
# Streamlit UI for File Upload and Visualization
# ------------------------------------------------------------------------------

uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')

    if st.button("Predict"):
        with st.spinner("Processing..."):
            predicted_label, pred_probs, waveform, sr = predict_class(uploaded_file)

        # Calculate confidence and additional metrics.
        confidence = np.max(pred_probs) * 100
        duration = len(waveform) / sr
        rms = np.sqrt(np.mean(waveform ** 2))

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

        # Compute a PCEN spectrogram with the specified parameters.
        pcen_spec = librosa.core.pcen(mel_spec, time_constant=0.007, power=0.85,
                                      bias=5.5, gain=0.7, eps=1e-08)

        # Plot both spectrograms side by side.
        fig_spec, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        img1 = librosa.display.specshow(log_mel_spec, sr=sr, x_axis='time', y_axis='mel', ax=ax1)
        ax1.set_title("Log Mel Spectrogram")
        fig_spec.colorbar(img1, ax=ax1, format="%+2.0f dB")

        # For the PCEN spectrogram, force the amplitude scale from 0 to 3.
        img2 = librosa.display.specshow(pcen_spec, sr=sr, x_axis='time', y_axis='mel',
                                        ax=ax2, cmap='magma', norm=plt.Normalize(vmin=0, vmax=3))
        ax2.set_title("PCEN Spectrogram")
        fig_spec.colorbar(img2, ax=ax2, format="%.2f")

        st.pyplot(fig_spec)

        # ------------------------------------------------------------------------------
        # Detailed Explanation Section
        # ------------------------------------------------------------------------------
        st.markdown("""
        ### Spectrogram Comparison and PCEN Advantages

        The spectrograms above illustrate a **comparison** between:

        - **Log-Mel Spectrogram:**  
          Derived by converting the power mel spectrogram into decibel (dB) units, this representation provides a clear view of the frequency content over time. However, it tends to be sensitive to background noise and amplitude variations.

        - **PCEN Spectrogram:**  
          This representation applies **Per-Channel Energy Normalization (PCEN)** to the mel spectrogram. PCEN dynamically normalizes the energy in each frequency band, suppressing background noise and enhancing transient features such as subtle lung sounds. Notice that the PCEN spectrogram is normalized (with values scaled between 0 and 3), indicating its effectiveness in reducing noise.

        **Model Classification Limitations:**  
        Our current model is limited to classifying lung sounds into **10 diagnostic categories** due to the relatively small dataset available. This limitation means that only conditions like Asthma, Bronchiectasis, Bronchiolitis, COPD, Healthy, Heart Failure, Lung Fibrosis, Pleural Effusion, Pneumonia, and URTI can be recognized. With a larger dataset, we could expand the model to recognize a broader range of respiratory conditions and provide more detailed diagnoses.

        **How Our Model Uses PCEN:**  
        By leveraging the PCEN spectrogram as input, our model focuses on diagnostically relevant features while minimizing the effects of ambient noise. The enhanced clarity of the PCEN spectrogram—compared to the traditional log-mel spectrogram—demonstrates its advantages in noisy clinical environments.

        This comparison highlights why PCEN is a crucial component of our lung sound classification pipeline.
        """)


#
# import streamlit as st
# import numpy as np
# import librosa
# import librosa.display
# import tensorflow as tf
# import os
# import tempfile
# import matplotlib.pyplot as plt
# import pandas as pd
#
# # ------------------------------------------------------------------------------
# # About This Project
# # ------------------------------------------------------------------------------
#
# st.title('🩺 Lung Sound Classification App')
#
# st.markdown("""
# **About This Project**
#
# This app is part of our research exploring advanced audio preprocessing techniques for lung sound classification, with a special focus on **Per-Channel Energy Normalization (PCEN)**. Traditional methods like log-mel spectrograms and MFCCs are often sensitive to noise, whereas PCEN adapts dynamically to suppress background noise and amplify subtle lung sounds.
#
# **Key Highlights:**
# - **Robust Feature Extraction:** PCEN effectively enhances transient respiratory sounds and minimizes background interference.
# - **Optimized Parameter Tuning:** Our experiments fine-tuned parameters (e.g., temporal smoothing *T*, dynamic range compression *r*, bias, and gain) to achieve superior performance on various datasets.
# - **Extensive Data Augmentation:** We segmented recordings into uniform 6-second clips and applied multiple augmentations to improve model generalization.
# - **Comprehensive Evaluation:** Our study evaluated models using metrics such as loss, accuracy, F1-score, and AUC_ROC, demonstrating PCEN’s consistent advantage.
#
# For more details on our methodology and results, please refer to our original research paper. You can also view our full project on [GitHub](https://github.com/IKnowUCantPvp/Lung-Sound-Classification-PCEN.git).
# """)
#
# st.write(
#     "Upload an audio file below to classify the lung sound based on our trained model."
# )
#
#
#
# # ------------------------------------------------------------------------------
# # Model Loading and Helper Functions
# # ------------------------------------------------------------------------------
#
# # Load your pre-trained model from the repository.
# # The model directory should contain the saved_model.pb file and related assets.
# MODEL_DIR = "Conv2DOldPCEN"
# model = tf.saved_model.load(MODEL_DIR)
# inference_func = model.signatures["serving_default"]
#
# # Mapping from class indices to human-readable labels.
# CLASS_NAMES = {
#     0: "Asthma",
#     1: "Bronchiectasis",
#     2: "Bronchiolitis",
#     3: "COPD",
#     4: "Healthy",
#     5: "Heart Failure",
#     6: "Lung Fibrosis",
#     7: "Pleural Effusion",
#     8: "Pneumonia",
#     9: "URTI",
# }
#
# def load_audio_waveform(file_path):
#     """
#     Loads an audio file, resamples it to 8 kHz, and adjusts its duration to 6 seconds.
#     Returns the raw waveform (1D numpy array) and the sampling rate.
#     """
#     target_sr = 8000  # 8 kHz sample rate.
#     target_duration = 6  # 6 seconds duration.
#     target_length = target_sr * target_duration  # 48000 samples.
#
#     y, sr = librosa.load(file_path, sr=target_sr)
#     if len(y) < target_length:
#         y = np.pad(y, (0, target_length - len(y)), mode='constant')
#     else:
#         y = y[:target_length]
#     return y, sr
#
# def preprocess_audio(file_path):
#     """
#     Loads the audio and returns a tensor of shape (1, 48000, 1)
#     which matches the model's expected input.
#     """
#     y, _ = load_audio_waveform(file_path)
#     y = np.expand_dims(y, axis=-1)  # Shape becomes (48000, 1)
#     return y[np.newaxis, ...]       # Final shape: (1, 48000, 1)
#
# def predict_class(audio_file):
#     """
#     Writes the uploaded audio file to a temporary file, preprocesses it,
#     and uses the loaded model to predict the lung sound classification.
#     Also returns the raw waveform and sampling rate for further visualization.
#     """
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
#         tmp.write(audio_file.read())
#         tmp_path = tmp.name
#
#     # Load audio for visualization.
#     waveform, sr = load_audio_waveform(tmp_path)
#     # Preprocess the audio for model input.
#     input_data = preprocess_audio(tmp_path)
#     os.remove(tmp_path)
#
#     input_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)
#     prediction = inference_func(input_tensor)
#     pred_tensor = list(prediction.values())[0]
#     pred_probs = pred_tensor.numpy()[0]
#     predicted_index = np.argmax(pred_probs)
#     predicted_label = CLASS_NAMES.get(predicted_index, "Unknown")
#     return predicted_label, pred_probs, waveform, sr
#
# # ------------------------------------------------------------------------------
# # Streamlit UI for File Upload and Visualization
# # ------------------------------------------------------------------------------
#
# uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg"])
#
# if uploaded_file is not None:
#     st.audio(uploaded_file, format='audio/wav')
#
#     if st.button("Predict"):
#         with st.spinner("Processing..."):
#             predicted_label, pred_probs, waveform, sr = predict_class(uploaded_file)
#
#         # Calculate confidence and additional metrics.
#         confidence = np.max(pred_probs) * 100
#         duration = len(waveform) / sr
#         rms = np.sqrt(np.mean(waveform ** 2))
#
#         st.success(f"Predicted lung sound classification: **{predicted_label}**")
#         st.write(f"Model Confidence: **{confidence:.2f}%**")
#         st.write(f"**Audio Duration:** {duration:.2f} seconds")
#         st.write(f"**Sampling Rate:** {sr} Hz")
#         st.write(f"**RMS Amplitude:** {rms:.4f}")
#
#         # Display probability distribution as a bar chart.
#         df = pd.DataFrame({
#             "Probability": pred_probs
#         }, index=list(CLASS_NAMES.values()))
#         st.write("### Probability Distribution for All Classes")
#         st.bar_chart(df)
#
#         # Plot the audio waveform.
#         fig_wave, ax_wave = plt.subplots(figsize=(10, 4))
#         times = np.linspace(0, duration, num=len(waveform))
#         ax_wave.plot(times, waveform)
#         ax_wave.set_title("Audio Waveform")
#         ax_wave.set_xlabel("Time (s)")
#         ax_wave.set_ylabel("Amplitude")
#         st.pyplot(fig_wave)
#
#         # Compute a log mel spectrogram.
#         n_fft = 2048
#         hop_length = 512
#         n_mels = 128
#         mel_spec = librosa.feature.melspectrogram(y=waveform, sr=sr, n_fft=n_fft,
#                                                   hop_length=hop_length, n_mels=n_mels)
#         log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
#
#         # Compute a PCEN spectrogram with the specified parameters.
#         pcen_spec = librosa.core.pcen(mel_spec, time_constant=0.007, power=0.85,
#                                       bias=5.5, gain=0.7, eps=1e-08)
#
#         # Plot both spectrograms side by side.
#         fig_spec, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
#
#         img1 = librosa.display.specshow(log_mel_spec, sr=sr, x_axis='time', y_axis='mel', ax=ax1)
#         ax1.set_title("Log Mel Spectrogram")
#         fig_spec.colorbar(img1, ax=ax1, format="%+2.0f dB")
#
#         # For the PCEN spectrogram, force the amplitude scale from 0 to 3.
#         img2 = librosa.display.specshow(pcen_spec, sr=sr, x_axis='time', y_axis='mel',
#                                         ax=ax2, cmap='magma', norm=plt.Normalize(vmin=0, vmax=3))
#         ax2.set_title("PCEN Spectrogram")
#         fig_spec.colorbar(img2, ax=ax2, format="%.2f")
#
#         st.pyplot(fig_spec)
