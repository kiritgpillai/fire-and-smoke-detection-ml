import streamlit as st

# Title
st.title("Fire and Smoke Detection")

# About section
st.header("About")
st.markdown(
    """
    Using Computer Vision to detect fire and smoke from Images. Every hallway or room nowadays has a camera in it, 
    especially in a country like Singapore where cameras are omnipresent. It thus seems redundant to have smoke and fire 
    detectors for the same regions, except for areas where cameras are not installed due to privacy concerns. This project aims to offer a reliable system to remove this redundancy.
    """
)

# Dataset section
st.markdown("---")
st.header("Data Processing")
st.write(
    """
    **Dataset Synopsis:** The dataset, an integral component of our "Fire and Smoke Detection" project, was meticulously sourced from a plethora of online repositories, including Kaggle and GitHub. The compilation consists of a variety of images, each representing diverse scenarios of fire and smoke phenomena. This collection comprises unlabeled images, with each folder indicating the occurrence of fire or smoke. The images are in JPEG format.

    **Data Visualization:** To gain instant insights from our dataset, we employed a range of visualization techniques. These visual aids provided an intuitive grasp of the data's characteristics and structure.

    **Data Preprocessing:** During the initial data preprocessing phase, we took steps to ensure the reliability and validity of our dataset. We proactively identified and eliminated spurious images from our data pool to avoid potential pitfalls during model training.

    **Feature Analysis:** We conducted a comprehensive feature analysis to elucidate the intricate relationships between various features within the dataset. This deep-dive investigation helped us understand the interdependencies between variables and informed our subsequent modeling strategies.
    """
)

# Project by section
st.markdown("---")
st.header("Project by:")
st.write("Kirit Govindaraja Pillai")
st.write("Shiv Gupta")
st.write("Muktansh")
st.write("Adrija")
st.write("Mahima")