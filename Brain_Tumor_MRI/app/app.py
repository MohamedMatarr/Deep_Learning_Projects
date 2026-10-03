import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

MODEL_PATH = Path(__file__).parent.parent / "Models" / "brain_tumor_model.h5"

model = tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

model = tf.keras.models.load_model(MODEL_PATH)


st.title("🧠 Brain Tumor MRI Classifier")
st.write("Upload an MRI image to predict the tumor class.")


uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded MRI",
        width=400
    )

    image = image.resize((224, 224))

    image = np.array(image).astype("float32") / 255.0

    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image, verbose=0)

    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.subheader("Prediction")

    st.success(predicted_class)

    st.write(f"Confidence: **{confidence:.2f}%**")