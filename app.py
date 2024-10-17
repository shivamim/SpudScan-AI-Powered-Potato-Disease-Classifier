import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os

# Set page configuration (optional)
st.set_page_config(page_title="Potato Disease Classifier", page_icon="🥔", layout="wide")

# Debugging: List contents of the current directory and the model directory
st.write("Contents of current directory:", os.listdir('.'))
if os.path.exists('model'):
    st.write("Contents of model directory:", os.listdir('model'))
else:
    st.write("Model directory not found")

# Load the model
@st.cache_resource
def load_model():
    model_path = 'model'
    
    print(f"Looking for model at: {model_path}")  # Debugging print
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model directory not found at {model_path}")
    
    try:
        options = tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
        model = tf.saved_model.load(model_path, options=options)
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

try:
    model = load_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load model: {str(e)}")
    st.stop()

# Define the class labels
class_names = ['Healthy', 'Early Blight', 'Late Blight']  # Adjust this list to match your model's output

# CSS styles
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

# Application header
st.markdown("<h1>Potato Disease Classifier 🥔</h1>", unsafe_allow_html=True)
st.write("Upload a potato leaf image to classify the disease.")

# Function to preprocess the uploaded image
def preprocess_image(image_data):
    image = ImageOps.fit(image_data, (256, 256), Image.LANCZOS)
    image = np.asarray(image)
    image = image.astype(np.float32) / 255.0  # Normalize the image to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Prediction function
def predict(image):
    processed_image = preprocess_image(image)
    infer = model.signatures["serving_default"]
    
    # Get predictions from the model
    predictions = infer(tf.constant(processed_image))
    
    # Use the correct output tensor name 'dense_5'
    output_tensor = predictions['dense_5']  # Use the output layer name here
    
    return output_tensor

# File uploader and prediction logic
uploaded_file = st.file_uploader("Choose an image of a potato leaf...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Show the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image and make prediction
    try:
        predictions = predict(image)
        
        # Calculate probabilities and get the predicted class
        probabilities = tf.nn.softmax(predictions[0]).numpy()
        predicted_class_index = tf.argmax(probabilities).numpy()
        predicted_class = class_names[predicted_class_index]
        confidence_score = probabilities[predicted_class_index] * 100  # Convert to percentage

        # Show the result
        st.markdown("<h2>Prediction Result</h2>", unsafe_allow_html=True)
        st.markdown(f"<div class='result-box'>Predicted Disease: <b>{predicted_class}</b></div>", unsafe_allow_html=True)
        st.write(f"Confidence Score: {confidence_score:.2f}%")  # Display confidence score as a percentage
    
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")

# Footer (Optional)
st.markdown("""
    <div style='text-align: center; margin-top: 50px;'>
        <small>Powered by TensorFlow & Streamlit</small>
    </div>
""", unsafe_allow_html=True)
