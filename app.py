import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import base64

# Set page configuration (optional)
st.set_page_config(page_title="Potato Disease Classifier", page_icon="🥔", layout="wide")

# Load the model
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('model')  # Adjust the path accordingly
    return model

model = load_model()

# Define the class labels
class_names = ['Healthy', 'Early Blight', 'Late Blight']  # Adjust this list to match your model's output

# CSS to make the app more interactive and visually appealing
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    h1 {
        color: #ff6347;
        text-align: center;
    }
    .upload-btn-wrapper {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .upload-btn-wrapper button {
        background-color: #008CBA;
        color: white;
        padding: 10px 20px;
        font-size: 16px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
    }
    .upload-btn-wrapper input[type=file] {
        font-size: 16px;
        padding: 10px;
    }
    .result-box {
        background-color: #ffebcd;
        padding: 20px;
        border-radius: 5px;
        font-size: 18px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# JavaScript for interactivity (e.g., scroll to result)
st.markdown("""
    <script>
    function scrollToResults() {
        document.getElementById("results").scrollIntoView();
    }
    </script>
""", unsafe_allow_html=True)

# Application header
st.markdown("<h1>Potato Disease Classifier 🥔</h1>", unsafe_allow_html=True)
st.write("Upload a potato leaf image to classify the disease.")

# Function to preprocess the uploaded image
def preprocess_image(image_data):
    image = ImageOps.fit(image_data, (256, 256), Image.ANTIALIAS)
    image = np.asarray(image)
    image = image.astype(np.float32) / 255.0  # Normalize the image to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# File uploader and prediction logic
uploaded_file = st.file_uploader("Choose an image of a potato leaf...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Show the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image and make prediction
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    predicted_class = class_names[np.argmax(predictions)]

    # Show the result with a button to scroll to it
    st.markdown("<div id='results'></div>", unsafe_allow_html=True)
    st.markdown("<h2>Prediction Result</h2>", unsafe_allow_html=True)
    st.markdown(f"<div class='result-box'>Predicted Disease: <b>{predicted_class}</b></div>", unsafe_allow_html=True)

    # Scroll to result button
    st.button("See Prediction", on_click=lambda: st.script("scrollToResults()"))

# Footer (Optional)
st.markdown("""
    <div style='text-align: center; margin-top: 50px;'>
        <small>Powered by TensorFlow & Streamlit</small>
    </div>
""", unsafe_allow_html=True)
