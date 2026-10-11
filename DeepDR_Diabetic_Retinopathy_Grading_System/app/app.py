import os
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# 1. Page Configuration (Set to 'wide' for a better Dashboard look)
st.set_page_config(page_title="DeepDR: Diabetic Retinopathy", page_icon="👁️", layout="wide")

# 2. Load Model with Caching
@st.cache_resource
def load_dr_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, '../model/resnet50v2_model.keras')
    model = load_model(model_path,compile=False, safe_mode=False)
    return model

try:
    model = load_dr_model()
    model_loaded = True
except Exception as e:
    st.error(f"Error loading model: {e}")
    model_loaded = False

# 3. Define Class Names (Fixed to match the model's alphabetical training order)
class_names = [
    'Mild', 
    'Moderate', 
    'No_DR', 
    'Proliferate_DR', 
    'Severe'
]

# 4. UI Design
st.title("👁️ DeepDR: Diabetic Retinopathy Grading System")
st.write("Upload a retina fundus image to predict the severity of Diabetic Retinopathy.")

# 5. Create two columns for the wide layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model_loaded:
    with col1:
        # Display the uploaded image in the left column
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_container_width=True)
    
    with col2:
        st.subheader("Analysis Results")
        st.write("🔍 Analyzing image...")
        
        try:
            # 6. Image Preprocessing
            if image.mode != "RGB":
                image = image.convert("RGB")
                
            img_resized = image.resize((224, 224))
            img_array = np.array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            
            # 7. Prediction
            predictions = model.predict(img_array)
            predicted_class_index = np.argmax(predictions, axis=1)[0]
            confidence = np.max(predictions) * 100
            
            # 8. Display Results in the right column
            st.markdown("---")
            st.success(f"**Diagnosis:** {class_names[predicted_class_index]}")
            st.info(f"**Confidence Level:** {confidence:.2f}%")
            
            # Optional: Show probabilities for all classes
            st.markdown("### Class Probabilities:")
            for i, class_name in enumerate(class_names):
                st.write(f"- **{class_name}:** {predictions[0][i] * 100:.2f}%")
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
