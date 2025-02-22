import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import os
import tempfile

st.title('🩺 Lung Sound Classification App')
st.write(
    "This app is built off of a research project exploring the usage of Per Channel Energy Normalization "
    "to classify lung sounds. Upload an audio file below to classify it!"
)

# Load your pre-trained model from the repository.
# The model directory should contain the saved_model.pb file and related assets.
MODEL_DIR = r"C:\Users\natha\OneDrive\Documents\GitHub\Lung-Class-Website\models\FinalModels\clean dataset\Conv2DOldPCEN"  # Adjust this relative path as needed.
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

def preprocess_audio(file_path):
    """
    Loads an audio file, resamples it to 8 kHz, and adjusts its duration to 6 seconds.
    Returns the raw waveform in the shape (1, 48000, 1) so that it matches the training input.
    """
    target_sr = 8000              # 8 kHz sample rate.
    target_duration = 6           # 6 seconds duration.
    target_length = target_sr * target_duration  # 48000 samples.

    # Load the audio file with the target sampling rate.
    y, _ = librosa.load(file_path, sr=target_sr)

    # Pad with zeros if the audio is shorter than 6 seconds, or trim if longer.
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)), mode='constant')
    else:
        y = y[:target_length]

    # Add channel dimension to match the model's expected input shape.
    y = np.expand_dims(y, axis=-1)  # Now shape is (48000, 1)
    # Add batch dimension (now shape becomes (1, 48000, 1))
    return y[np.newaxis, ...]

def predict_class(audio_file):
    """
    Writes the uploaded audio file to a temporary file, preprocesses it,
    and uses the loaded model to predict the lung sound classification.
    """
    # Save the uploaded audio file temporarily.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_file.read())
        tmp_path = tmp.name

    # Preprocess the audio file to obtain the raw waveform.
    waveform = preprocess_audio(tmp_path)
    os.remove(tmp_path)

    # Convert the waveform to a TensorFlow tensor.
    input_tensor = tf.convert_to_tensor(waveform, dtype=tf.float32)

    # Run inference using the model's serving function.
    prediction = inference_func(input_tensor)
    # Extract the output tensor (assumes a single output tensor is returned).
    pred_tensor = list(prediction.values())[0]

    # Identify the predicted class by taking the argmax.
    predicted_index = np.argmax(pred_tensor.numpy(), axis=1)[0]
    predicted_label = CLASS_NAMES.get(predicted_index, "Unknown")
    return predicted_label

# Streamlit UI for file upload and prediction.
st.write("Upload an audio file to classify the lung sound:")
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg"])

if uploaded_file is not None:
    # Display an audio player for the uploaded file.
    st.audio(uploaded_file, format='audio/wav')

    if st.button("Predict"):
        with st.spinner("Processing..."):
            result = predict_class(uploaded_file)
        st.success(f"Predicted lung sound classification: **{result}**")
