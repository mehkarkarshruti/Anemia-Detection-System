# Anemia Detection System

A deep learning application to detect anemia from 
microscopic blood smear images using MobileNetV2.

## What it does
Upload a microscopic blood smear image and the model 
classifies it as Anemic or Healthy with a confidence score.

## Dataset
Trained on the AneRBC dataset — 9,600 training images 
and 2,400 validation images.

## Model Performance
- Accuracy: 81.71%
- Precision: 84% (Anemic), 80% (Healthy)
- Recall: 78% (Anemic), 85% (Healthy)

## Tech Stack
Python, TensorFlow, Keras, MobileNetV2, OpenCV, Streamlit

## How to run

pip install -r requirements.txt
streamlit run app.py


Upload a microscopic blood smear image for best results.
Camera input is also supported but microscopic images 
give more accurate results.

## Note
This is for educational purposes only. 
Not a substitute for medical diagnosis.
