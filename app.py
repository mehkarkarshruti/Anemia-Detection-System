import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from io import BytesIO

# Set page configuration
st.set_page_config(
    page_title="Anemia Detection System",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #e63946;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #1d3557;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        background-color: #f8f9fa;
        border-left: 5px solid #e63946;
    }
    .confidence-high {
        color: #2a9d8f;
        font-weight: bold;
    }
    .confidence-medium {
        color: #e9c46a;
        font-weight: bold;
    }
    .confidence-low {
        color: #e63946;
        font-weight: bold;
    }
    .stButton>button {
        background-color: #e63946;
        color: white;
        font-weight: bold;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #d62828;
    }
    .info-box {
        background-color: #a8dadc;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">🩸 Anemia Detection System</h1>', unsafe_allow_html=True)
st.markdown("""
This application uses a deep learning model to classify whether a person is **anemic** or **healthy** based on microsopic blood sample images(RBC).

### How it works:
1. Upload an image of the microscopic blood sample
2. The model processes the image using MobileNetV2 architecture
3. Get instant classification results with confidence scores
""")

# Sidebar for additional information and controls
with st.sidebar:
    st.markdown("## ℹ️ About This Tool")
    st.markdown("""
    This system uses a trained convolutional neural network (CNN) based on MobileNetV2 to analyze microscopic blood sample images.
    
    **Model Performance:**
    - Accuracy: 81.71%
    - Precision: 84% (Anemic), 80% (Healthy)
    - Recall: 78% (Anemic), 85% (Healthy)
    
    **Important Note:**
    This tool is for **educational and screening purposes only**. Always consult with a healthcare professional for medical diagnosis.
    """)
    
    st.markdown("---")
    
    st.markdown("### 📊 Model Information")
    st.markdown("""
    **Architecture:** MobileNetV2 with custom layers
    **Input Size:** 224×224 pixels
    **Classes:** Anemic (0), Healthy (1)
    **Training Data:** 9,600 images
    **Validation Data:** 2,400 images
    """)
    
    st.markdown("---")
    
    # Model loading indicator
    st.markdown("### 🔄 Model Status")
    model_status = st.empty()
    model_status.info("Loading model...")

# Load model with caching
@st.cache_resource
def load_anemia_model():
    try:
        model = keras.models.load_model('final_model.h5')
        return model, "✅ Model loaded successfully!"
    except Exception as e:
        try:
            model = keras.models.load_model('best_model.h5')
            return model, "✅ Model loaded successfully!"
        except:
            return None, f"❌ Error loading model: {str(e)}"

# Initialize session state for storing results
if 'results' not in st.session_state:
    st.session_state.results = []

# Load the model
model, model_message = load_anemia_model()
model_status.success(model_message)

# Image preprocessing function
def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocess the uploaded image for model prediction
    """
    # Convert PIL Image to numpy array
    image_array = np.array(image)
    
    # Convert to RGB if grayscale
    if len(image_array.shape) == 2:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
    elif image_array.shape[2] == 4:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
    
    # Resize image
    image_array = cv2.resize(image_array, target_size)
    
    # Normalize pixel values
    image_array = image_array.astype('float32') / 255.0
    
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

# Create tabs for different functionalities
tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload Image", "📸 Take Photo", "📊 Model Insights", "ℹ️ About Anemia"])

with tab1:
    st.markdown('<h3 class="sub-header">Upload an Image for Analysis</h3>', unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Upload an image of the microscopic blood sample image"
    )
    
    if uploaded_file is not None:
        # Display the uploaded image
        col1, col2 = st.columns(2)
        
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Add image information
            st.info(f"**Image Details:**\n- Format: {image.format}\n- Size: {image.size}\n- Mode: {image.mode}")
        
        with col2:
            if model is not None:
                # Preprocess and predict
                with st.spinner("Analyzing image..."):
                    processed_image = preprocess_image(image)
                    prediction = model.predict(processed_image, verbose=0)
                    probability = float(prediction[0][0])
                    
                    # Determine classification
                    if probability > 0.5:
                        classification = "Healthy"
                        confidence = probability
                        confidence_class = "healthy"
                    else:
                        classification = "Anemic"
                        confidence = 1 - probability
                        confidence_class = "anemic"
                    
                    # Confidence level styling
                    if confidence > 0.8:
                        confidence_level = "high"
                        confidence_color = "confidence-high"
                    elif confidence > 0.6:
                        confidence_level = "medium"
                        confidence_color = "confidence-medium"
                    else:
                        confidence_level = "low"
                        confidence_color = "confidence-low"
                    
                    # Display results
                    st.markdown('<div class="result-box">', unsafe_allow_html=True)
                    st.markdown(f"### 🩺 Diagnosis Result")
                    st.markdown(f"**Classification:** **{classification}**")
                    st.markdown(f"**Confidence:** <span class='{confidence_color}'>{confidence*100:.2f}% ({confidence_level})</span>", unsafe_allow_html=True)
                    st.markdown(f"**Raw Probability:** {probability:.4f}")
                    
                    # Interpretation
                    if classification == "Anemic":
                        st.warning("""
                        **Recommendation:** 
                        - Consult with a healthcare professional
                        - Consider blood tests for confirmation
                        - Monitor for symptoms like fatigue, pale skin, or dizziness
                        """)
                    else:
                        st.success("""
                        **Recommendation:** 
                        - Maintain healthy iron-rich diet
                        - Continue regular health checkups
                        - Monitor for any changes in symptoms
                        """)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Store result in session state
                    result_entry = {
                        "image": uploaded_file.name,
                        "classification": classification,
                        "confidence": confidence,
                        "probability": probability,
                        "timestamp": st.session_state.get('timestamp', 'N/A')
                    }
                    st.session_state.results.append(result_entry)
                    
                    # Visualization
                    fig, ax = plt.subplots(figsize=(8, 4))
                    categories = ['Anemic', 'Healthy']
                    probabilities = [1 - probability, probability]
                    colors = ['#e63946', '#2a9d8f']
                    
                    bars = ax.bar(categories, probabilities, color=colors, alpha=0.8)
                    ax.set_ylabel('Probability', fontsize=12)
                    ax.set_title('Classification Probabilities', fontsize=14, fontweight='bold')
                    ax.set_ylim([0, 1])
                    
                    # Add value labels on bars
                    for bar, prob in zip(bars, probabilities):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                               f'{prob:.1%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
                    
                    st.pyplot(fig)
                    
                    # Download results button
                    result_text = f"""Anemia Detection Result
Image: {uploaded_file.name}
Classification: {classification}
Confidence: {confidence*100:.2f}%
Probability: {probability:.4f}
Date: {st.session_state.get('timestamp', 'N/A')}

Note: This result is for screening purposes only. Please consult a healthcare professional for medical diagnosis.
"""
                    
                    st.download_button(
                        label="📥 Download Results",
                        data=result_text,
                        file_name="anemia_detection_result.txt",
                        mime="text/plain"
                    )
            else:
                st.error("Model is not available. Please check if the model file exists.")

with tab2:
    st.markdown('<h3 class="sub-header">Capture Image Using Camera</h3>', unsafe_allow_html=True)
    
    st.markdown("""
    **Instructions for capturing blood sample images:**
    1. Ensure good lighting conditions
    2. Capture clear image without blur
    """)
    
    # Camera input
    camera_image = st.camera_input("Take a picture of the blood sample")
    
    if camera_image is not None:
        if model is not None:
            # Process camera image
            with st.spinner("Analyzing captured image..."):
                image = Image.open(camera_image)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(image, caption="Captured Image", use_column_width=True)
                
                with col2:
                    processed_image = preprocess_image(image)
                    prediction = model.predict(processed_image, verbose=0)
                    probability = float(prediction[0][0])
                    
                    if probability > 0.5:
                        classification = "Healthy"
                        confidence = probability
                    else:
                        classification = "Anemic"
                        confidence = 1 - probability
                    
                    # Display results
                    st.markdown('<div class="result-box">', unsafe_allow_html=True)
                    st.markdown(f"### 📸 Camera Analysis Result")
                    st.markdown(f"**Classification:** **{classification}**")
                    st.markdown(f"**Confidence:** **{confidence*100:.2f}%**")
                    
                    if classification == "Anemic":
                        st.warning("Screening suggests possible anemia. Please consult a healthcare provider.")
                    else:
                        st.success("Screening suggests normal results.")
                    
                    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<h3 class="sub-header">Model Performance and Insights</h3>', unsafe_allow_html=True)
    
    if model is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("###  Model Architecture")
            st.markdown("""
            **Base Model:** MobileNetV2 (pretrained on ImageNet)
            **Custom Layers:**
            - Global Average Pooling 2D
            - Batch Normalization
            - Dense Layer (128 neurons, ReLU)
            - Dropout (0.5)
            - Output Layer (1 neuron, Sigmoid)
            """)
            
            st.markdown("###  Training Details")
            st.markdown("""
            **Dataset:** 9,600 training images, 2,400 validation images
            **Classes:** Anemic vs Healthy
            **Image Size:** 224×224 pixels
            **Augmentation:** Rotation, shifts, zoom, flip
            **Optimizer:** Adam (1e-4 learning rate)
            """)
        
        with col2:
            st.markdown("###  Performance Metrics")
            
            # Create metrics display
            metrics_col1, metrics_col2 = st.columns(2)
            
            with metrics_col1:
                st.metric(label="Accuracy", value="81.71%")
                st.metric(label="Precision (Anemic)", value="84%")
            
            with metrics_col2:
                st.metric(label="Recall (Healthy)", value="85%")
                st.metric(label="F1-Score", value="81-82%")
            
            st.markdown("### 📊 Confusion Matrix")
            st.markdown("""
            | | Predicted Anemic | Predicted Healthy |
            |---|---|---|
            | **Actual Anemic** | 937 | 263 |
            | **Actual Healthy** | 176 | 1024 |
            """)
            
            st.markdown("""
            **Interpretation:**
            - **Sensitivity (Anemic):** 78.08%
            - **Specificity (Healthy):** 85.33%
            """)
        
        st.markdown("---")
        st.markdown("### 📋 Recent Predictions")
        
        if st.session_state.results:
            # Display recent results
            for i, result in enumerate(reversed(st.session_state.results[-5:]), 1):
                st.markdown(f"""
                **Prediction {i}:** {result['image']}
                - Result: {result['classification']}
                - Confidence: {result['confidence']*100:.1f}%
                """)
        else:
            st.info("No predictions made yet. Upload an image to see results here.")
        
        # Clear results button
        if st.button("Clear Prediction History"):
            st.session_state.results = []
            st.success("Prediction history cleared!")
            st.rerun()

with tab4:
    st.markdown('<h3 class="sub-header">Understanding Anemia</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ℹ️ What is Anemia?")
        st.markdown("""
        Anemia is a condition where you lack enough healthy red blood cells to carry adequate oxygen to your body's tissues.
        
        **Common Symptoms:**
        - Fatigue and weakness
        - Pale or yellowish skin
        - Irregular heartbeats
        - Shortness of breath
        - Dizziness or lightheadedness
        - Chest pain
        - Cold hands and feet
        """)
        
        st.markdown("### 🔍 How it's Diagnosed")
        st.markdown("""
        Traditional diagnosis methods:
        1. **Complete Blood Count (CBC)**
        2. **Iron studies**
        3. **Vitamin B12 and folate tests**
        4. **Physical examination** (including conjunctiva inspection)
        """)
    
    with col2:
        
        st.markdown("### ⚠️ Important Disclaimer")
        st.markdown("""
        <div class="info-box">
        **Medical Disclaimer:**
        This tool is designed for educational and screening purposes only. It is NOT a substitute for professional medical advice, diagnosis, or treatment.
        
        **Always:**
        - Consult with qualified healthcare professionals
        - Follow up with proper medical tests
        - Never disregard professional medical advice
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📚 References")
        st.markdown("""
        1. World Health Organization. (2021). Anaemia.
        2. Clinical assessment of pallor in anemia diagnosis
        3. Deep learning applications in medical imaging
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6c757d; font-size: 0.9rem;">
    <p>Anemia Detection System v1.0 | For Educational Purposes | Always consult healthcare professionals for medical diagnosis</p>
    <p>Model trained on blood smear images using TensorFlow and MobileNetV2</p>
</div>
""", unsafe_allow_html=True)

# Add timestamp to session state
if 'timestamp' not in st.session_state:
    from datetime import datetime
    st.session_state.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")