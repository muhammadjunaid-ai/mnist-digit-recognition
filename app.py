import os
# Silence TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# 1. Load the model using your specific filename
MODEL_NAME = 'mnist_model.h5'

try:
    # Load and re-compile to avoid the 'metrics' warning in logs
    model = tf.keras.models.load_model(MODEL_NAME)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

def predict_digit(data):
    # Step A: Extract image from Gradio dictionary
    image_array = data['composite'] if isinstance(data, dict) else data
    
    # Step B: Convert to PIL and Grayscale
    img = Image.fromarray(image_array.astype('uint8')).convert("L")
    
    # Step C: INVERT COLORS 
    # This turns your black ink into white pixels (what MNIST expects)
    img = ImageOps.invert(img)
    
    # Step D: Resize to 28x28
    img = img.resize((28, 28))
    
    # Step E: Prepare for Model
    img_array = np.array(img) / 255.0  # Normalization
    img_reshaped = img_array.reshape(1, 28, 28, 1) # Add batch and channel dims
    
    # Step F: Predict
    prediction = model.predict(img_reshaped, verbose=0)
    
    # Prepare the confidence scores for the top 3 classes
    results = {str(i): float(prediction[0][i]) for i in range(10)}
    
    # Return both the label and the processed image for visual confirmation
    return results, img

# 2. Build the Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# MNIST Digit Recognition")
    gr.Markdown("Draw a digit (0-9) clearly in the box below.")
    
    with gr.Row():
        with gr.Column():
            input_canvas = gr.Sketchpad(label="Draw Here", type="numpy")
            submit_btn = gr.Button("Predict", variant="primary")
            
        with gr.Column():
            output_label = gr.Label(num_top_classes=3, label="Predictions")
            preview_img = gr.Image(label="What the model sees (28x28 inverted)", image_mode="L")

    submit_btn.click(
        fn=predict_digit, 
        inputs=input_canvas, 
        outputs=[output_label, preview_img]
    )

# 3. Launch
if __name__ == "__main__":
    demo.launch()
