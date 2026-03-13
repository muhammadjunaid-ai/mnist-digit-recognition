# MNIST Digit Recognition App

## Project Overview
This project is a deep learning web application that recognizes handwritten digits (0–9). Users can draw a digit on an interactive canvas, and the trained model predicts the number in real time. The application is built using Python, TensorFlow/Keras, NumPy, and Gradio, and it is deployed on Hugging Face Spaces.

## Live Demo
Try the application here:  
https://muhammadjunaidai-mnist-digit-app.hf.space

## How the Application Works
The user draws a digit on the sketchpad canvas. The application processes the image by converting it to grayscale, inverting the colors, resizing it to 28×28 pixels, and normalizing the pixel values. The processed image is then passed to a trained deep learning model that predicts the digit. The application also shows the processed 28×28 image so users can see exactly what the model receives.

## Features
The application allows users to draw digits (0–9) directly on a canvas and receive real-time predictions. It displays the top predicted classes with confidence scores and shows the processed 28×28 inverted image that the model uses for prediction.

## Dataset
The model is trained on the MNIST dataset, which contains 70,000 grayscale images of handwritten digits. Each image is 28×28 pixels and represents digits from 0 to 9. This dataset is widely used for training and evaluating machine learning and deep learning models.

## Technologies Used
This project is built using Python, TensorFlow/Keras for the deep learning model, NumPy for numerical operations, Pillow for image processing, and Gradio for creating the interactive web interface. The application is deployed on Hugging Face Spaces.

## Project Structure
app.py – Main application code for the Gradio interface  
mnist_model.h5 – Trained deep learning model  
requirements.txt – Python dependencies  
README.md – Project documentation

## Running the Project Locally
Clone the repository:

git clone https://github.com/muhammadjunaid-ai/mnist-digit-recognition

Navigate to the project folder:

cd mnist-digit-recognition

Install dependencies:

pip install -r requirements.txt

Run the application:

python app.py

## Author
Muhammad Junaid  
BS Artificial Intelligence  
GitHub: https://github.com/muhammadjunaid-ai

