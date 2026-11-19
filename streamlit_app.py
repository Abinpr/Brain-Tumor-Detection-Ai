import io
import os
import sqlite3
import uuid
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image, ImageStat

# ----------------------------- Configuration -----------------------------
st.set_page_config(
    page_title="NeuroScan AI | Brain MRI Research",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "Brain_Tumor_Final_Model.keras")
DB_PATH = os.path.join(APP_DIR, "analysis_history.db")

CLASS_NAMES = [
    "Glioma Tumor",
    "No Tumor",
    "Meningioma Tumor",
    "Pituitary Tumor",
]
CLASS_KEYS = ["glioma_tumor", "no_tumor", "meningioma_tumor", "pituitary_tumor"]

st.markdown("""
<style>
.stApp { background: #080d19; color: #e8eef8; }
section[data-testid="stSidebar"] { background: #101a2b; border-right: 1px solid #26364f; }
.block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1400px; }
.hero { padding: 1.5rem 1.7rem; border: 1px solid #263b59; border-radius: 18px;
        background: linear-gradient(120deg,#13233a,#0d1728); margin-bottom: 1.2rem; }
.hero h1 { color: #79d7ff; margin: 0; font-size: 2.2rem; }
.hero p { color: #b8c8dc; margin: .4rem 0 0 0; }
.metric-card { background:#111e31; border:1px solid #263b59; border-radius:14px; padding:1rem; }
.small-muted { color:#9fb0c7; font-size:.88rem; }
div[data-testid="stMetric"] { background:#111e31; border:1px solid #263b59; padding:14px; border-radius:12px; }
.stButton>button { border-radius:10px; }
</style>
""", unsafe_allow_html=True)

# ----------------------------- Persistence ------------------------------
def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            case_label TEXT,
            predicted_class TEXT NOT NULL,
            confidence REAL NOT NULL,
            probabilities TEXT NOT NULL,
            width INTEGER,
            height INTEGER,
            brightness REAL,
            contrast REAL,
            image_quality TEXT
        )
    """)
    conn.commit()
    return conn

conn = get_connection()

def save_analysis(record):
    conn.execute(
        """INSERT INTO analyses
        (id, created_at, case_label, predicted_class, confidence, probabilities,
         width, height, brightness, contrast, image_quality)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            record["id"], record["created_at"], record["case_label"],
            record["predicted_class"], record["confidence"],
            ",".join(f"{x:.8f}" for x in record["probabilities"]),
            record["width"], record["height"], record["brightness"],
            record["contrast"], record["image_quality"],
        ),
    )
    conn.commit()

def load_history():
    return pd.read_sql_query(
        "SELECT id, created_at, case_label, predicted_class, confidence, width, height, brightness, contrast, image_quality FROM analyses ORDER BY created_at DESC",
        conn,
    )

# ----------------------------- Model -------------------------------------
@st.cache_resource(show_spinner="Loading the trained EfficientNetB0 model...")
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. Keep Brain_Tumor_Final_Model.keras beside streamlit_app.py."
        )
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

def predict_image(pil_image):
    model = load_model()
    rgb = np.asarray(pil_image.convert("RGB"))
    # Match the original training pipeline: RGB -> BGR, resize to 150x150.
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    resized = cv2.resize(bgr, (150, 150), interpolation=cv2.INTER_AREA)
    batch = np.expand_dims(resized.astype(np.float32), axis=0)
    probs = np.asarray(model.predict(batch, verbose=0)[0], dtype=float)
    probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    total = probs.sum()
    if total > 0:
        probs = probs / total
    index = int(np.argmax(probs))
    return index, probs, bgr

# ----------------------------- Image analysis ----------------------------
def image_metrics(pil_image):
    rgb = np.asarray(pil_image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if width < 128 or height < 128:
        quality = "Low resolution"
    elif laplacian_variance < 30:
        quality = "Potentially blurry"
    else:
        quality = "Basic checks passed"
    return {
        "width": width,
        "height": height,
        "brightness": brightness,
        "contrast": contrast,
        "sharpness": laplacian_variance,
        "quality": quality,
        "file_size_kb": None,
    }

# ----------------------------- Grad-CAM ----------------------------------
def find_last_conv_layer(model):
    found = []
    def walk(layer, prefix=""):
        for child in getattr(layer, "layers", []):
            name = f"{prefix}{child.name}"
            if isinstance(child, tf.keras.layers.Conv2D):
                found.append((child, name))
            if getattr(child, "layers", None):
                walk(child, name + "/")
    walk(model)
    return found[-1] if found else (None, None)

def make_gradcam(model, bgr_image, class_index):
    conv_layer, layer_name = find_last_conv_layer(model)
    if conv_layer is None:
        raise ValueError("No Conv2D layer was found in this model.")
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[conv_layer.output, model.output],
    )
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (150, 150), interpolation=cv2.INTER_AREA)
    input_tensor = np.expand_dims(resized.astype(np.float32), axis=0)
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(input_tensor, training=False)
        score = predictions[:, class_index]
    grads = tape.gradient(score, conv_outputs)
    if grads is None:
        raise ValueError("Gradients could not be calculated for the selected layer.")
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    feature_map = conv_outputs[0]
    heatmap = tf.reduce_sum(feature_map * weights, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    max_value = tf.reduce_max(heatmap)
    heatmap = heatmap / (max_value + tf.keras.backend.epsilon())
    heatmap = cv2.resize(heatmap.numpy(), (bgr_image.shape[1], bgr_image.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap)
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(bgr_image, 0.58, colored, 0.42, 0)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), layer_name

# ----------------------------- PDF report --------------------------------
def create_pdf_report(record, metrics):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=18*mm, leftMargin=18*mm,
                            topMargin=18*mm, bottomMargin=18*mm)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Note", parent=styles["BodyText"], textColor=colors.HexColor("#555555"), leading=14))
    story = [
        Paragraph("Brain MRI AI Analysis Report", styles["Title"]),
        Spacer(1, 8),
        Paragraph("Educational research prototype report. Not a medical diagnosis.", styles["Note"]),
        Spacer(1, 14),
    ]
    rows = [
        ["Field", "Result"],
        ["Analysis ID", record["id"]],
        ["Date and time", record["created_at"]],
        ["Case label", record["case_label"] or "Not provided"],
        ["Model prediction", record["predicted_class"]],
        ["Model score", f'{record["confidence"]:.2f}%'],
        ["Image dimensions", f'{metrics["width"]} × {metrics["height"]} pixels'],
        ["Brightness (mean gray)", f'{metrics["brightness"]:.2f}'],
        ["Contrast (gray standard deviation)", f'{metrics["contrast"]:.2f}'],
        ["Image quality check", metrics["quality"]],
    ]
    table = Table(rows, colWidths=[58*mm, 105*mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#16324f")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), .4, colors.HexColor("#c5cfda")),
        ("BACKGROUND", (0,1), (-1,-1), colors.HexColor("#f4f7fa")),
        ("PADDING", (0,0), (-1,-1), 7),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
    ]))
    story.extend([table, Spacer(1, 14), Paragraph("Class probability scores", styles["Heading2"])])
    prob_rows = [["Class", "Score"]]
    for name, value in zip(CLASS_NAMES, record["probabilities"]):
        prob_rows.append([name, f"{value*100:.2f}%"])
    prob_table = Table(prob_rows, colWidths=[90*mm, 73*mm])
    prob_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#16324f")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), .4, colors.HexColor("#c5cfda")),
        ("PADDING", (0,0), (-1,-1), 7),
    ]))
    story.extend([prob_table, Spacer(1, 18),
                  Paragraph("Important limitation: This model has not been clinically validated. "
                            "Scores represent model outputs, not the probability that a patient has a disease. "
                            "Do not use this report to make healthcare decisions. Consult a qualified medical professional.",
                            styles["Note"])])
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

# ----------------------------- Session state -----------------------------
if "current_record" not in st.session_state:
    st.session_state.current_record = None
if "current_image" not in st.session_state:
    st.session_state.current_image = None
if "current_bgr" not in st.session_state:
    st.session_state.current_bgr = None
if "current_metrics" not in st.session_state:
    st.session_state.current_metrics = None

# ----------------------------- Sidebar -----------------------------------
st.sidebar.markdown("## 🧠 NeuroScan AI")
st.sidebar.caption("Brain MRI research and classification prototype")
pages = [
    "🏠 Dashboard",
    "🧠 MRI Detection",
    "🔬 MRI Analysis",
    "🎯 Tumor Localization",
    "📊 Analysis Results",
    "📄 PDF Reports",
    "👤 Patient History",
    "🤖 AI Assistant",
    "ℹ️ About Project",
]
page = st.sidebar.radio("NAVIGATION", pages, label_visibility="collapsed")
st.sidebar.divider()
st.sidebar.caption("For educational use only. Not for clinical diagnosis.")

st.markdown("""
<div class="hero">
<h1>🧠 NeuroScan AI</h1>
<p>Deep-learning-based brain MRI image classification and research dashboard</p>
</div>
""", unsafe_allow_html=True)

# ----------------------------- Dashboard ---------------------------------
if page == "🏠 Dashboard":
    history = load_history()
    total = len(history)
    st.subheader("Project dashboard")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Saved analyses", total)
    c2.metric("Tumor categories", "3")
    c3.metric("Model input", "150 × 150")
    c4.metric("Architecture", "EfficientNetB0")
    st.markdown("### Explore the application")
    tiles = st.columns(3)
    descriptions = [
        ("🧠 MRI Detection", "Upload an MRI image and run the four-class model."),
        ("🔬 MRI Analysis", "Review image dimensions, brightness, contrast and sharpness."),
        ("🎯 Tumor Localization", "View Grad-CAM regions that influenced a model output."),
        ("📊 Analysis Results", "Compare the model's class scores."),
        ("📄 PDF Reports", "Export the latest analysis summary."),
        ("👤 Patient History", "Review saved case records on this app instance."),
        ("🤖 AI Assistant", "Read educational answers about brain tumor concepts."),
        ("ℹ️ About Project", "Review model details, evaluation and limitations."),
    ]
    for i, (title, desc) in enumerate(descriptions):
        with tiles[i % 3]:
            st.markdown(f'<div class="metric-card"><b>{title}</b><p class="small-muted">{desc}</p></div>', unsafe_allow_html=True)
    if total:
        st.markdown("### Recent activity")
        st.dataframe(history.head(5), use_container_width=True, hide_index=True)
    else:
        st.info("No analyses saved yet. Open MRI Detection to run your first analysis.")

# ----------------------------- MRI Detection ----------------------------
elif page == "🧠 MRI Detection":
    st.subheader("Upload and classify an MRI image")
    st.caption("Accepted formats: JPG, JPEG and PNG. The model resizes images to 150 × 150 pixels.")
    case_label = st.text_input("Optional case label", placeholder="Example: Case 001 (avoid entering personal identifiers)")
    uploaded = st.file_uploader("Choose an MRI image", type=["jpg", "jpeg", "png"], key="mri_upload")
    if uploaded:
        try:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption="Uploaded MRI image", use_container_width=True)
            if st.button("🔍 Analyze MRI image", type="primary", use_container_width=True):
                with st.spinner("Running model inference..."):
                    idx, probs, bgr = predict_image(image)
                    metrics = image_metrics(image)
                    record = {
                        "id": str(uuid.uuid4())[:8].upper(),
                        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "case_label": case_label.strip(),
                        "predicted_class": CLASS_NAMES[idx],
                        "confidence": float(probs[idx] * 100),
                        "probabilities": probs.tolist(),
                        "width": metrics["width"],
                        "height": metrics["height"],
                        "brightness": metrics["brightness"],
                        "contrast": metrics["contrast"],
                        "image_quality": metrics["quality"],
                    }
                    save_analysis(record)
                    st.session_state.current_record = record
                    st.session_state.current_image = image
                    st.session_state.current_bgr = bgr
                    st.session_state.current_metrics = metrics
                st.success("Analysis completed and saved to history.")
                st.metric("Model's top-scoring class", record["predicted_class"], f'{record["confidence"]:.2f}% model score')
                st.warning("This is an experimental model output, not a diagnosis. A high score does not establish disease presence.")
        except Exception as exc:
            st.error(f"Could not process this image: {exc}")

# ----------------------------- MRI Analysis ------------------------------
elif page == "🔬 MRI Analysis":
    st.subheader("MRI image quality and properties")
    uploaded = st.file_uploader("Upload an MRI image", type=["jpg", "jpeg", "png"], key="quality_upload")
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        m = image_metrics(image)
        a, b = st.columns([1, 1])
        with a:
            st.image(image, caption="Image under review", use_container_width=True)
        with b:
            st.metric("Resolution", f'{m["width"]} × {m["height"]}')
            st.metric("Brightness", f'{m["brightness"]:.2f} / 255')
            st.metric("Contrast", f'{m["contrast"]:.2f}')
            st.metric("Sharpness indicator", f'{m["sharpness"]:.2f}')
            st.info(f'Basic quality check: {m["quality"]}')
        st.caption("These simple computer-vision measurements are not a radiology quality assessment and do not determine whether an MRI is medically suitable.")

# ----------------------------- Tumor Localization ------------------------
elif page == "🎯 Tumor Localization":
    st.subheader("Grad-CAM visualization")
    record = st.session_state.current_record
    bgr = st.session_state.current_bgr
    if record is None or bgr is None:
        st.info("Run an MRI analysis first in the MRI Detection section. Its image and prediction are used here.")
    else:
        st.write(f"Current model output: **{record['predicted_class']}** ({record['confidence']:.2f}% score)")
        if st.button("Generate Grad-CAM heatmap", type="primary"):
            try:
                model = load_model()
                class_index = CLASS_NAMES.index(record["predicted_class"])
                overlay, layer_name = make_gradcam(model, bgr, class_index)
                st.image(overlay, caption=f"Grad-CAM overlay | layer: {layer_name}", use_container_width=True)
                st.caption("Warm regions indicate areas that contributed more strongly to this model output. They are not tumor boundaries or a confirmed tumor location.")
            except Exception as exc:
                st.error(f"Grad-CAM could not be generated for this model graph: {exc}")
                st.info("The prediction feature remains available. Grad-CAM needs a compatible connected convolutional layer.")

# ----------------------------- Results -----------------------------------
elif page == "📊 Analysis Results":
    record = st.session_state.current_record
    if record is None:
        st.info("Run an MRI analysis first to see its probability chart.")
    else:
        st.subheader("Prediction score breakdown")
        st.metric("Top-scoring class", record["predicted_class"], f'{record["confidence"]:.2f}%')
        df = pd.DataFrame({"Class": CLASS_NAMES, "Model score (%)": np.array(record["probabilities"]) * 100})
        fig = px.bar(df, x="Class", y="Model score (%)", color="Class", text_auto=".2f",
                     title="Model output scores by class")
        fig.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e8eef8")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.warning("These are model scores, not calibrated clinical probabilities.")

# ----------------------------- PDF Reports ------------------------------
elif page == "📄 PDF Reports":
    st.subheader("Download an analysis report")
    record = st.session_state.current_record
    metrics = st.session_state.current_metrics
    if record is None or metrics is None:
        st.info("Complete an MRI analysis first. The latest result will be used for the report.")
    else:
        st.write(f"Report for analysis **{record['id']}**")
        try:
            pdf_bytes = create_pdf_report(record, metrics)
            st.download_button(
                "⬇️ Download PDF report",
                data=pdf_bytes,
                file_name=f"brain_mri_report_{record['id']}.pdf",
                mime="application/pdf",
                type="primary",
            )
        except Exception as exc:
            st.error(f"Could not create PDF report: {exc}")

# ----------------------------- History -----------------------------------
elif page == "👤 Patient History":
    st.subheader("Saved analysis history")
    st.caption("Records are stored in a local SQLite file on the app server. Cloud app storage may be temporary and is not a secure medical-record system.")
    history = load_history()
    if history.empty:
        st.info("No saved analyses yet.")
    else:
        st.dataframe(history, use_container_width=True, hide_index=True)
        csv_bytes = history.to_csv(index=False).encode("utf-8")
        st.download_button("Download history as CSV", csv_bytes, "analysis_history.csv", "text/csv")
        selected = st.selectbox("Select a record to inspect", history["id"].tolist())
        row = history[history["id"] == selected].iloc[0]
        st.json(row.to_dict())
        if st.button("Delete selected history record"):
            conn.execute("DELETE FROM analyses WHERE id = ?", (selected,))
            conn.commit()
            st.success("Record deleted.")
            st.rerun()

# ----------------------------- Educational assistant --------------------
elif page == "🤖 AI Assistant":
    st.subheader("Brain tumor educational assistant")
    st.caption("A small built-in FAQ assistant. It does not diagnose, interpret scans or replace a clinician.")
    faq = {
        "What is a brain tumor?": "A brain tumor is an abnormal growth of cells in or around the brain. Tumors can be benign or malignant, and their effects depend on type, size and location.",
        "What is glioma?": "Gliomas are tumors that arise from glial cells or related precursor cells in the central nervous system. They include several distinct tumor types.",
        "What is meningioma?": "Meningiomas arise from the meninges, the membranes surrounding the brain and spinal cord. Many grow slowly, but behavior varies.",
        "What is pituitary tumor?": "Pituitary tumors develop in the pituitary gland. Many are benign and may affect hormone production or nearby structures.",
        "What does no tumor mean?": "In this project's dataset labels, 'no tumor' is one of the model's four classes. A model output of 'no tumor' does not rule out disease or replace a radiologist's report.",
        "What is MRI?": "Magnetic resonance imaging uses magnetic fields and radio waves to create detailed images of internal body structures, including the brain.",
        "What is Grad-CAM?": "Grad-CAM uses gradients flowing into convolutional feature maps to create a heatmap of image regions that influenced a neural network's class score. It is an explanation aid, not a segmentation or tumor boundary.",
        "Can this app diagnose cancer?": "No. This student research prototype is not clinically validated. Do not use it for diagnosis, treatment or emergency decisions. Consult a qualified healthcare professional.",
    }
    question = st.selectbox("Choose a question", list(faq.keys()))
    if st.button("Get educational answer"):
        st.info(faq[question])
    custom = st.text_input("Or enter a topic to search in the FAQ", placeholder="Example: Grad-CAM")
    if custom:
        matches = [v for k, v in faq.items() if custom.lower() in k.lower()]
        if matches:
            st.write(matches[0])
        else:
            st.caption("No exact FAQ match. Try: MRI, glioma, meningioma, pituitary, Grad-CAM, or diagnosis.")

# ----------------------------- About -------------------------------------
elif page == "ℹ️ About Project":
    st.subheader("About this project")
    st.write("""
    **NeuroScan AI** is an academic deep-learning prototype that classifies uploaded brain MRI images into four dataset categories:
    glioma tumor, no tumor, meningioma tumor and pituitary tumor.

    **Model:** EfficientNetB0-based image classifier with a four-unit softmax output layer.  
    **Input pipeline:** RGB image converted to BGR, resized to 150 × 150 pixels, then passed to the saved model.  
    **Frameworks:** Python, TensorFlow/Keras, Streamlit, OpenCV, NumPy, Pillow, SQLite and Plotly.
    """)
    st.markdown("### Recorded test-set evaluation")
    st.caption("Metrics below are from the project's previously recorded evaluation. They should be checked against the final notebook before formal submission.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Test accuracy", "81.98%")
    c2.metric("Macro precision", "0.87")
    c3.metric("Macro recall", "0.82")
    st.markdown("### Important limitations")
    st.warning("""
    - The model is not clinically validated and must not be used for medical diagnosis.
    - The recorded test recall for the glioma class was low (0.36), so some glioma examples were missed.
    - Dataset performance does not guarantee performance on images from other hospitals, scanners or populations.
    - Grad-CAM is an approximate explanation map, not a tumor segmentation.
    - The built-in history database is for demonstration only and is not a secure, persistent patient-record system.
    """)
    st.markdown("### Future improvements")
    st.write("External validation, larger and more diverse datasets, class-wise performance improvements, secure authentication, encrypted persistent storage, radiologist review and clinical validation.")

st.divider()
st.caption("NeuroScan AI • Academic project • EfficientNetB0 + TensorFlow • Not for clinical use")
