import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# Load the trained model
model = tf.keras.models.load_model('violence_detection.keras')

# Define class labels
class_labels = ['Non-Violent', 'Violent']

# Streamlit title and description
st.title("Violence Detection from Images")
st.write("Upload an image, and the model will predict whether the scene is violent or not.")

# Function to preprocess the uploaded image
def preprocess_image(image):
    img = np.array(image)
    img = cv2.resize(img, (32, 32))  # Resize to match model's input size
    img = img.astype('float32') / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# File uploader for images
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Predict the class
    prediction = model.predict(processed_image)
    predicted_class = class_labels[np.argmax(prediction)]

    # Display the prediction
    st.write(f"Prediction: {predicted_class}")
