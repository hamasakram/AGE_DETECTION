import streamlit as st
import pandas as pd
import numpy as np
import os
from PIL import Image
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

def load_and_prepare_data(data_csv, data_dir, samples_per_class=100):
    data = pd.read_csv(data_csv)
    images = []
    labels = []
    
    class_data = {
        'YOUNG': data[data['Class'] == 'YOUNG'].sample(n=samples_per_class, random_state=42),
        'MIDDLE': data[data['Class'] == 'MIDDLE'].sample(n=samples_per_class, random_state=42),
        'OLD': data[data['Class'] == 'OLD'].sample(n=samples_per_class, random_state=42)
    }
    
    for category, df in class_data.items():
        for _, row in df.iterrows():
            image_path = os.path.join(data_dir, row['ID'])
            try:
                img = Image.open(image_path).resize((128, 128))
                images.append(np.array(img))
                labels.append(category)
            except IOError:
                print(f"Error: Cannot open {image_path}")
    
    return np.array(images), labels

def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def predict_image(uploaded_image, model):
    try:
        img = Image.open(uploaded_image).convert('RGB').resize((128, 128))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)
        predictions = model.predict(img)
        predicted_class_index = np.argmax(predictions, axis=1)
        class_labels = ['YOUNG', 'MIDDLE', 'OLD']
        return class_labels[predicted_class_index[0]]
    except Exception as e:
        return str(e)

# Streamlit app starts here
st.title('Age Detection Model')

# Upload the image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Load model (assuming the model has already been trained and saved)
    model_path = 'age_detection_model.h5'
    model = load_model(model_path)

    # Predict the class of the uploaded image
    result = predict_image(uploaded_image, model)
    st.write(f"Predicted Class: {result}")
else:
    st.write("Please upload an image to classify.")

