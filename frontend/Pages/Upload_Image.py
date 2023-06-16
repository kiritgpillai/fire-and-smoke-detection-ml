import os
import time
import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import pandas as pd
import matplotlib.pyplot as plt

# Load the model from .pth file, mapping it to the CPU
model = torch.load('model_final.pth', map_location=torch.device('cpu'))

# Set the model to evaluation mode
model.eval()

st.write("""
         # Fire and Smoke Detection
         """
         )
st.write("This is an image classification web app to predict fire and smoke from the image uploaded by the user")

file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

class_names = ['Fire', 'Neutral', 'Smoke']

def predict(image):
    prediction_transform = transforms.Compose([transforms.Resize(size=(224, 224)),
                                               transforms.ToTensor(),
                                               transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    image = prediction_transform(image)[:3, :, :].unsqueeze(0)

    pred = model(image)
    probabilities = torch.softmax(pred, dim=1)[0] * 100
    return probabilities

def accu_predict(image):
    prediction_transform = transforms.Compose([transforms.Resize(size=(224, 224)),
                                               transforms.ToTensor(),
                                               transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    image = prediction_transform(image)[:3, :, :].unsqueeze(0)

    pred = model(image)
    idx = torch.argmax(pred)
    prob = pred[0][idx].item() * 100
    return class_names[idx], prob


if file is None:
    st.text("")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    with st.spinner("Predicting..."):
        probabilities = predict(image)
        accuracy = accu_predict(image)
        time.sleep(3)
        st.success("Done!")

    time.sleep(2)
    predicted_class = class_names[probabilities.argmax()]
    confidence_percentage = probabilities.max()

    st.write(f"Predicted Class: {predicted_class}")
    st.write(f"Accuracy: {accuracy[1]:.2f}%")
    # st.write(f"Confidence Percentage: {confidence_percentage:.2f}%")


#For plotting the pie chart

    # Create a DataFrame with the class labels and percentages
    data = {
        'Class': class_names,
        'Percentage': probabilities.tolist()
    }
    df = pd.DataFrame(data)

    # Filter out the classes with 0 percentage
    filtered_df = df[df['Percentage'] > 0]
    fig, ax = plt.subplots()
    wedges, text, autotext = ax.pie(filtered_df['Percentage'], labels=filtered_df['Class'], autopct='%1.1f%%')
    ax.set_title('Analysis',color='white')
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle

    # Set the chart background color
    fig.patch.set_facecolor('#10141c')
    # Set the text color of chart text to white
    for autotext in autotext:
        autotext.set_color('white')
    for text in text:
        text.set_color('white')
    st.pyplot(fig)
