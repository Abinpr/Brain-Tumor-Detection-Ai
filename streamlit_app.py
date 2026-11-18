import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Brain Tumor Detection AI",
    page_icon="🧠",
    layout="wide"
)

# Custom styling
st.markdown("""
<style>
.stApp {
    background-color: #0b1020;
    color: white;
}
.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #64d8ff;
}
.subtitle {
    text-align: center;
    color: #b5c5d8;
    font-size: 17px;
}
.result-box {
    padding: 20px;
    border-radius: 15px;
    background-color: #17243b;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "Brain_Tumor_Final_Model.keras"
    )

model = load_model()

# Class names
labels = [
    "Glioma Tumor",
    "No Tumor",
    "Meningioma Tumor",
    "Pituitary Tumor"
]

# Header
st.markdown(
    '<div class="main-title">🧠 Brain Tumor Detection AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Deep Learning Based MRI Image Classification</div>',
    unsafe_allow_html=True
)

st.write("")
st.divider()

# Upload section
st.subheader("Upload MRI Scan")

uploaded_file = st.file_uploader(
    "Choose an MRI image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            image,
            caption="Uploaded MRI Image",
            use_container_width=True
        )

    with col2:
        st.markdown("### Analysis")

        if st.button("Analyze MRI Image", use_container_width=True):

            img = np.array(image)

            img = cv2.cvtColor(
                img,
                cv2.COLOR_RGB2BGR
            )

            img = cv2.resize(img, (150, 150))

            img = np.expand_dims(img, axis=0)

            predictions = model.predict(img, verbose=0)[0]

            predicted_index = np.argmax(predictions)

            confidence = predictions[predicted_index] * 100

            st.success("Analysis completed!")

            st.markdown(
                f"""
                <div class="result-box">
                    <h3>Predicted Result</h3>
                    <h2>{labels[predicted_index]}</h2>
                    <h3>Confidence: {confidence:.2f}%</h3>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.write("### Class Probabilities")

            for i, label in enumerate(labels):
                st.write(
                    f"{label}: {predictions[i] * 100:.2f}%"
                )

                st.progress(float(predictions[i]))

st.divider()

st.warning(
    "Educational prototype only. This model is not clinically "
    "validated and must not be used for medical diagnosis."
)

st.caption("Brain Tumor Detection Project | EfficientNetB0 | TensorFlow")