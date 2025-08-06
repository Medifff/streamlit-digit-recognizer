import streamlit as st
import torch
import torch.nn as nn
from torchvision.transforms import ToTensor
from PIL import Image
import numpy as np
from streamlit_drawable_canvas import st_canvas

# Set the page configuration
st.set_page_config(layout="wide", page_title="Digit Recognizer")

# --- 1. Define the Neural Network Architecture ---
# This MUST be the exact same architecture as the one we trained.
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# --- 2. Load the Trained Model ---
# Use a caching decorator to load the model only once.
@st.cache_resource
def load_model(model_path):
    """Loads the pre-trained PyTorch model."""
    model = NeuralNetwork()
    # Load the saved state dictionary.
    # map_location='cpu' ensures the model loads on the CPU, which is important
    # for broad compatibility in a web app.
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval() # Set the model to evaluation mode
    return model

# --- 3. Image Processing Function ---
def process_image(image_data):
    """Converts the canvas drawing to the format the model expects."""
    # Convert the canvas data to a PIL Image
    img = Image.fromarray(image_data.astype('uint8'), 'RGBA')
    # Convert to grayscale
    img = img.convert('L')
    # Resize to 28x28 pixels
    img = img.resize((28, 28))
    # Invert colors (MNIST is white on black, canvas is black on white)
    img = Image.fromarray(255 - np.array(img))
    # Convert to a PyTorch tensor and add batch and channel dimensions
    tensor = ToTensor()(img).unsqueeze(0)
    return tensor

# --- Main App ---
st.title("Handwritten Digit Recognizer 🧠")
st.write("Draw a digit from 0 to 9 on the canvas below, and the AI will try to guess what it is.")

# Load the model
try:
    model = load_model("mnist_model.pth")
except FileNotFoundError:
    st.error("Model file 'mnist_model.pth' not found. Please run the training script first to create it.")
    st.stop()

# --- User Interface ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Drawing Canvas")
    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=20,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=300,
        width=300,
        drawing_mode="freedraw",
        key="canvas",
    )

with col2:
    st.subheader("Prediction")
    if canvas_result.image_data is not None:
        # Check if the user has drawn something
        if canvas_result.image_data.any():
            # Process the image and get the model's prediction
            input_tensor = process_image(canvas_result.image_data)
            
            with torch.no_grad():
                prediction = model(input_tensor)
                # Use softmax to get probabilities
                probabilities = nn.functional.softmax(prediction[0], dim=0)
                predicted_digit = probabilities.argmax().item()
                confidence = probabilities.max().item()

            st.success(f"## I think it's a: **{predicted_digit}**")
            st.info(f"Confidence: **{confidence:.2%}**")

            # Display the probabilities in a bar chart
            st.write("Prediction Probabilities:")
            prob_df = pd.DataFrame({
                'Digit': list(range(10)),
                'Probability': probabilities.numpy()
            })
            st.bar_chart(prob_df.set_index('Digit'))
