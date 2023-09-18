import streamlit as st
import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
import tempfile
import os
from pathlib import Path

# Function to load and preprocess the uploaded image
def preprocess_image(image, save_directory):
    # Create the save directory if it doesn't exist
    os.makedirs(save_directory, exist_ok=True)

    # Save the uploaded image to the save directory
    image_path = os.path.join(save_directory, image.name)
    with open(image_path, "wb") as f:
        f.write(image.read())

    # Read and preprocess the image using OpenCV
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))  # Resize to the input size expected by your model
    img = img / 255.0  # Normalize pixel values to [0, 1]
    img = img.reshape(1, 224, 224, 3)  # Reshape for model input

    return img

# Load your trained model
model = keras.models.load_model(Path('artifacts/training/model.h5'))  # Replace with the path to your model

# Streamlit sidebar with information about pneumonia
st.sidebar.title("Pneumonia Information")
st.sidebar.write("Pneumonia is an inflammatory condition of the lung affecting primarily the tiny air sacs known as alveoli.")
st.sidebar.write("Common symptoms include fever, cough, and difficulty breathing.")
st.sidebar.write("Early detection and treatment are crucial for better outcomes.")

# Main Streamlit app
st.title("Pneumonia Detector")

# Upload an image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Specify the directory where you want to save the uploaded image
    save_directory = "uploaded_images"

    # Preprocess the image and save it to the specified directory
    processed_image = preprocess_image(uploaded_image, save_directory)

    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Make predictions when the user clicks the "Predict" button
    if st.button("Predict"):
        # Make a prediction
        prediction = model.predict(processed_image)

        # Determine the class label based on the prediction
        if prediction[0][0] > 0.5:
            st.error("Prediction: Pneumonia")
        else:
            st.success("Prediction: No Pneumonia")
