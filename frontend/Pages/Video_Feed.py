import cv2
import numpy as np
import streamlit as st
import torch
from torchvision.transforms import ToTensor
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('model_final.pth', map_location=device)
model.eval()

class_names = ['Fire', 'Smoke', 'Neutral']

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Initialize Streamlit
st.title('Fire and Smoke Detection')
st.write("This is an image classification web app to predict fire and smoke")

# Perform inference on the video feed
while True:
    # Get the frame from the video feed
    ret, frame = cap.read()

    # Preprocess the frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = ToTensor()(frame).unsqueeze(0).to(device)

    # Get the output of the model
    output = model(frame)

    # Calculate the predicted class and accuracy
    _, predicted_class = torch.max(output, 1)
    accuracy = torch.softmax(output, dim=1)[0, predicted_class] * 100

    # Get the class label
    predicted_class_label = class_names[predicted_class]

    # Display the video feed
    st.image(frame.squeeze(0).permute(1, 2, 0).cpu().numpy())

    # Display the accuracy and prediction
    st.write(f"Prediction: {predicted_class_label}")
    st.write(f"Accuracy: {accuracy.item():.2f}%")

    # Add a small delay between frame captures
    time.sleep(1.5)
