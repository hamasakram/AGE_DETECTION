# Age Detection Model

## Project Overview
This project implements an **Age Detection Model** using **Convolutional Neural Networks (CNN)** in TensorFlow and provides a **Streamlit** interface for predicting the age category of individuals based on uploaded images. The model classifies images into three categories:
- **YOUNG**
- **MIDDLE**
- **OLD**

## Features
- **Image Upload and Classification**: Users can upload an image to classify it into an age group.
- **Streamlit UI**: A clean, interactive interface for user-friendly predictions.
- **CNN Model**: A custom Convolutional Neural Network trained for age classification.

## Requirements
- Python 3.x
- TensorFlow
- Streamlit
- Pandas
- Numpy
- PIL (Pillow)

## Dataset
The dataset consists of labeled images classified into three categories: YOUNG, MIDDLE, and OLD. The images are resized to 128x128 pixels for uniformity. A sample of 100 images per class is used for training.

## Model Architecture
The Convolutional Neural Network (CNN) includes:

Convolutional Layers: Extract spatial features from images.
Batch Normalization: Stabilizes training and accelerates convergence.
Pooling Layers: Reduces spatial dimensions.
Dropout: Reduces overfitting.
Dense Layers: Classifies the image into the age categories.
# Usage
## Run the Streamlit app:
streamlit run age.py
1. Upload an image in .jpg, .png, or .jpeg format.
2. The app will display the predicted age category.
## Example Input and Output
Input: An image of a person.
Output: Age category (e.g., "YOUNG").
## Contributions
Contributions are welcome! Feel free to fork this repository, make improvements, and submit a pull request.
## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/hamasakram/AGE_DETECTION.git

 
