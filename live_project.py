import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib

# Set Streamlit page configuration
st.set_page_config(
    page_title="Iris Species Prediction",
    page_icon="😎",
    layout="centered"
)

# --- Load the trained model and scaler ---
@st.cache_resource
def load_assets():
    """Loads the trained model and StandardScaler from disk."""
    try:
        # Load the saved Keras model
        model = load_model('iris_model.h5')
        
        # Load the fitted scaler object
        scaler = joblib.load('scaler.pkl')
        
        # Define species mapping
        species_map = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}
        
        return model, scaler, species_map
    except Exception as e:
        st.error(f"Error loading required files (model.h5 or scaler.pkl): {e}")
        st.stop()

model, scaler, species_map = load_assets()

# --- Title and Description ---
st.title("Iris Species Prediction 😎")
st.write("Enter the measurements of the iris flower to predict its species.")
st.markdown("---")

# --- User Input Fields ---
st.header("Flower Measurements (in cm)")

# Use a form to group inputs and a single button
with st.form(key='iris_form'):
    col1, col2 = st.columns(2)
    with col1:
        sepal_length = st.number_input("Sepal Length", min_value=0.0, max_value=10.0, value=5.1)
        petal_length = st.number_input("Petal Length", min_value=0.0, max_value=10.0, value=1.4)
    with col2:
        sepal_width = st.number_input("Sepal Width", min_value=0.0, max_value=10.0, value=3.5)
        petal_width = st.number_input("Petal Width", min_value=0.0, max_value=10.0, value=0.2)

    st.markdown("---")
    submit_button = st.form_submit_button(label='Predict Species')

# --- Prediction Logic and Output ---
if submit_button:
    # Gather user inputs into a NumPy array
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    try:
        # Apply the same scaling used during training
        scaled_data = scaler.transform(input_data)
        
        # Get the model's prediction
        prediction_array = model.predict(scaled_data, verbose=0)
        predicted_class_index = np.argmax(prediction_array)
        predicted_species = species_map[predicted_class_index]
        confidence = prediction_array[0][predicted_class_index]

        # Display the result with a celebratory message
        st.success(f"### The predicted species is: **{predicted_species}**")
        st.write(f"Confidence: **{confidence:.2f}%**")
        st.balloons()

    except Exception as e:
        st.error(f"Prediction failed. Please check your model files. Error: {e}")

# --- Disclaimer ---
st.markdown("---")
st.markdown("Model trained on the famous Iris dataset.")