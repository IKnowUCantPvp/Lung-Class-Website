import streamlit as st

st.title('🩺Lung Sound Classificaiton App')

st.write('This app is built off of a research project researching the uasge of utilizing Per Channel Energy Normalization to classify lung sounds!')



# Load your pre-trained model from the repository.
# The model directory should contain the saved_model.pb file and related assets.
MODEL_DIR = "./my_lung_sound_model"  # Adjust this relative path as needed.
model = tf.saved_model.load(saved_model.pb)
# Get the default serving function from the model.
inference_func = model.signatures["serving_default"]

# Mapping from class indices to human-readable labels.
# Adjust these labels to match your model's training.
CLASS_NAMES = {
    0: "Asthma",
    1: "Bronchiectasis",
    2: "Bronchiolitis",
    3: "COPD",
    4: "Healthy"
    5: "Heart_Failure"
    6: "Lung_Fibrosis"
    7: "Pleural Effusion"
    8: "Pneumonia"
    9: "URTI"
}

def preprocess_audio(file_path):
    """
    Loads an audio file, resamples it to 8 kHz, and adjusts its duration to 6 seconds.
    Then extracts MFCC features for prediction.
    """
    target_sr = 8000              # Target sampling rate: 8 kHz.
    target_duration = 6           # Target duration: 6 seconds.
    target_length = target_sr * target_duration  # Total samples required.

    # Load the audio file with the target sampling rate.
    y, sr = librosa.load(file_path, sr=target_sr)
    
    # Pad with zeros if the audio is shorter than 6 seconds, or trim if longer.
    if len(y) < target_length:
        padding = target_length - len(y)
        y = np.pad(y, (0, padding), mode='constant')
    else:
        y = y[:target_length]
    
    # Extract MFCC features (using 40 coefficients as an example).
    mfccs = librosa.feature.mfcc(y=y, sr=target_sr, n_mfcc=40)
    
    # Compute the mean of MFCC coefficients over time to get a fixed-length vector.
    mfccs_mean = np.mean(mfccs.T, axis=0)
    
    # Reshape to match the model input shape (1, number_of_features).
    return mfccs_mean.reshape(1, -1)

def predict_class(audio_file):
    """
    Writes the uploaded audio file to a temporary file, preprocesses it,
    and uses the loaded model to predict the lung sound classification.
    """
    # Save the uploaded audio file temporarily.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_file.read())
        tmp_path = tmp.name

    # Preprocess the audio file to extract features.
    features = preprocess_audio(tmp_path)
    
    # Remove the temporary file.
    os.remove(tmp_path)

    # Convert features to a TensorFlow tensor.
    input_tensor = tf.convert_to_tensor(features, dtype=tf.float32)

    # Run the model inference. The result is returned as a dictionary.
    prediction = inference_func(input_tensor)
    # Extract the output tensor (assumes a single output tensor).
    pred_tensor = list(prediction.values())[0]
    
    # Identify the predicted class by taking the argmax.
    predicted_index = np.argmax(pred_tensor.numpy(), axis=1)[0]
    predicted_label = CLASS_NAMES.get(predicted_index, "Unknown")
    return predicted_label

# Streamlit UI
st.title("Lung Sound Classification")
st.write("Upload an audio file to classify the lung sound.")

uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg"])

if uploaded_file is not None:
    # Display an audio player for the uploaded file.
    st.audio(uploaded_file, format='audio/wav')
    
    if st.button("Predict"):
        with st.spinner("Processing..."):
            result = predict_class(uploaded_file)
        st.success(f"Predicted lung sound classification: **{result}**")
