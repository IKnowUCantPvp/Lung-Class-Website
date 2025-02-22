import streamlit as st

st.title('🩺Lung Sound Classificaiton App')

st.write('This app is built off of a research project researching the uasge of utilizing Per Channel Energy Normalization to classify lung sounds!')

# Load your pre-trained model.
# For example, if using a Keras model saved in an .h5 file:
MODEL_PATH = 'my_lung_sound_model.h5'
model = tf.keras.models.load_model(MODEL_PATH)

# Example mapping from class indices to human-readable labels.
# Update these labels based on your model's training.
CLASS_NAMES = {
    0: "Normal",
    1: "Crackles",
    2: "Wheezes",
    3: "Both",
    4: "Other"
}

def preprocess_audio(file_path):
    """
    Loads an audio file and extracts features (e.g., MFCCs) for prediction.
    Adjust the parameters as needed for your model.
    """
    # Load the audio file. Set sr (sample rate) to None to preserve original.
    y, sr = librosa.load(file_path, sr=None)
    
    # Extract MFCC features (using 40 coefficients as an example).
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    
    # Compute the mean of MFCC coefficients over time to get a fixed-length feature vector.
    mfccs_mean = np.mean(mfccs.T, axis=0)
    
    # Reshape to match model input shape (1, number_of_features)
    return mfccs_mean.reshape(1, -1)

def predict_class(audio_file):
    """
    Saves the uploaded audio file temporarily, preprocesses it,
    and uses the model to predict the lung sound classification.
    """
    # Write the uploaded file to a temporary file.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_file.read())
        tmp_path = tmp.name

    # Preprocess the audio file to extract features.
    features = preprocess_audio(tmp_path)

    # Clean up the temporary file.
    os.remove(tmp_path)

    # Get model prediction.
    prediction = model.predict(features)
    predicted_index = np.argmax(prediction, axis=1)[0]
    predicted_label = CLASS_NAMES.get(predicted_index, "Unknown")
    return predicted_label

# Streamlit UI
st.title("Lung Sound Classification")
st.write("Upload an audio file to predict the lung sound classification.")

uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    
    if st.button("Predict"):
        with st.spinner("Processing..."):
            result = predict_class(uploaded_file)
        st.success(f"Predicted lung sound classification: **{result}**")
