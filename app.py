"""
AI-Powered Visual Assistant for Visually Impaired People.
An accessible, multimodal Streamlit web application integrating YOLOv8 object detection,
EasyOCR text extraction, BLIP scene captioning, BLIP visual question answering,
and Whisper / gTTS speech interaction.
"""

import os
import sys
import io
import time
from typing import Optional
from PIL import Image
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st

from src.utils.device_manager import DeviceManager, get_cache_dir
from src.vision.object_detector import ObjectDetector
from src.ocr.ocr_reader import OCRReader
from src.nlp.scene_captioner import SceneCaptioner
from src.nlp.vqa_engine import VQAEngine
from src.speech.speech_recognizer import SpeechRecognizer
from src.speech.text_to_speech import TextToSpeech
from src.utils.context_synthesizer import ContextSynthesizer
from src.vision.utils import load_image, create_synthetic_test_image


# --- PAGE CONFIGURATION & ACCESSIBLE STYLING ---
st.set_page_config(
    page_title="AI Visual Assistant",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# High-contrast, screen-reader friendly CSS
ACCESSIBLE_CSS = """
<style>
    /* High contrast color scheme */
    .stApp {
        background-color: #121212;
        color: #F5F5F5;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Header typography */
    h1, h2, h3 {
        color: #FFD700 !important;
        font-weight: 700 !important;
    }
    
    /* Prominent callout cards */
    .hazard-card {
        background-color: #3b0d0d;
        border-left: 6px solid #FF3333;
        padding: 16px;
        border-radius: 8px;
        margin-bottom: 16px;
        font-size: 1.15rem;
        color: #FFFFFF;
    }
    
    .scene-card {
        background-color: #1e2638;
        border-left: 6px solid #3399FF;
        padding: 16px;
        border-radius: 8px;
        margin-bottom: 16px;
        font-size: 1.15rem;
        color: #FFFFFF;
    }
    
    .text-card {
        background-color: #102e1a;
        border-left: 6px solid #00FF66;
        padding: 16px;
        border-radius: 8px;
        margin-bottom: 16px;
        font-size: 1.15rem;
        color: #FFFFFF;
    }
    
    /* Action buttons */
    .stButton > button {
        background-color: #FFD700 !important;
        color: #000000 !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        padding: 12px 24px !important;
        border-radius: 8px !important;
        border: 2px solid #FFFFFF !important;
        min-height: 48px;
    }
    
    .stButton > button:hover {
        background-color: #FFEA70 !important;
        border-color: #FFD700 !important;
    }
    
    /* Accessibility focus indicators */
    button:focus, input:focus, select:focus {
        outline: 3px solid #00E5FF !important;
        outline-offset: 2px !important;
    }
    
    /* Audio player enlargement */
    audio {
        width: 100% !important;
        height: 50px !important;
        margin-top: 8px !important;
    }
</style>
"""
st.markdown(ACCESSIBLE_CSS, unsafe_allow_html=True)


# --- SINGLETON SERVICES INITIALIZATION (CACHED) ---
@st.cache_resource
def get_tts():
    return TextToSpeech()

@st.cache_resource
def get_speech_recognizer():
    return SpeechRecognizer(model_size="tiny")


# --- APPLICATION HEADER & SAFETY DISCLAIMER ---
st.title("👁️ AI-Powered Visual Assistant")
st.markdown("### Assistive Vision, OCR, and Natural-Language Question Answering")

# Safety banner
st.warning(
    "⚠️ **Academic Prototype Notice**: This application is strictly an academic research prototype "
    "and assistive cognitive aid. It **does NOT provide calibrated 3D depth perception** and **must NOT "
    "be used as an autonomous navigation system or for life-critical mobility**."
)


# --- SIDEBAR: HARDWARE MONITOR & SETTINGS ---
with st.sidebar:
    st.header("⚙️ Hardware & Diagnostics")
    
    vram = DeviceManager.get_vram_info()
    if vram["cuda_available"]:
        st.success(f"🎮 **GPU Active**: {vram['device_name']}")
        col_v1, col_v2 = st.columns(2)
        col_v1.metric("Allocated VRAM", f"{vram['allocated_mb']} MB")
        col_v2.metric("Free VRAM", f"{vram['free_vram_mb']} MB")
    else:
        st.info("💻 **Running on CPU Mode**")
    
    st.caption(f"📁 **Models Cache**: `{get_cache_dir()}`")
    
    if st.button("🧹 Clear GPU VRAM Cache", help="Evicts unused model allocations"):
        DeviceManager.clear_vram_cache()
        st.success("VRAM cache evicted!")
        st.rerun()

    st.markdown("---")
    st.header("🔊 Audio Settings")
    play_through_speakers = st.checkbox(
        "Play through laptop speakers (pyttsx3)", 
        value=False,
        help="Speaks aloud through your computer's built-in Windows SAPI5 audio"
    )
    auto_play_audio = st.checkbox("Auto-play browser audio", value=True)


# --- MAIN INTERACTION MODES ---
tabs = st.tabs([
    "🔍 Full Scene Analysis",
    "💬 Voice & Visual Q&A",
    "🎯 Object & Obstacle Detector",
    "📄 Sign & Text Reader",
    "ℹ️ About & Limitations"
])

tts = get_tts()


# =========================================================================
# TAB 1: FULL SCENE ANALYSIS (VISION + OCR + CAPTION + SPOKEN SYNTHESIS)
# =========================================================================
with tabs[0]:
    st.markdown("#### Analyze your complete visual environment")
    
    input_source = st.radio(
        "Select Camera or Image Input:",
        ["Upload Image File", "Use Webcam Camera", "Use Sample Street Scene"],
        horizontal=True
    )
    
    pil_image = None
    if input_source == "Upload Image File":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="full_upload")
        if uploaded_file is not None:
            pil_image = load_image(uploaded_file.read())
    elif input_source == "Use Webcam Camera":
        camera_photo = st.camera_input("Capture camera view", key="full_cam")
        if camera_photo is not None:
            pil_image = load_image(camera_photo.read())
    else:
        sample_path = os.path.join(PROJECT_ROOT, "sample_images", "bus.jpg")
        if os.path.exists(sample_path):
            pil_image = load_image(sample_path)
            st.info("Loaded sample street scene: bus.jpg")

    if pil_image is not None:
        col_left, col_right = st.columns([1, 1])
        with col_left:
            st.image(pil_image, caption="Visual View", use_container_width=True)

        with col_right:
            if st.button("⚡ Describe My Surroundings", type="primary", use_container_width=True):
                with st.spinner("Analyzing objects, text, and scene context..."):
                    # 1. YOLO
                    detector = ObjectDetector(model_name="yolov8n.pt", confidence_threshold=0.35)
                    detections = detector.detect(pil_image)
                    annotated_img = detector.annotate(pil_image, detections)
                    detector.unload()

                    # 2. EasyOCR
                    ocr = OCRReader(confidence_threshold=0.25)
                    ocr_results = ocr.read_text(pil_image)
                    ocr.unload()

                    # 3. BLIP Caption
                    captioner = SceneCaptioner()
                    caption = captioner.generate_caption(pil_image)
                    captioner.unload()

                    # 4. Synthesize
                    synthesis = ContextSynthesizer.synthesize(
                        caption=caption,
                        detections=detections,
                        ocr_results=ocr_results
                    )
                    spoken_text = synthesis["spoken_narrative"]

                # Render synthesized cards
                if synthesis["immediate_hazards"]:
                    st.markdown(
                        f"<div class='hazard-card'>🚨 <b>Obstacle Alert</b>: "
                        f"{', '.join(synthesis['immediate_hazards'])}</div>",
                        unsafe_allow_html=True
                    )

                if synthesis["scene_caption"]:
                    st.markdown(
                        f"<div class='scene-card'>🌐 <b>Scene Overview</b>: "
                        f"{synthesis['scene_caption'].capitalize()}</div>",
                        unsafe_allow_html=True
                    )

                if synthesis["ocr_text"]:
                    st.markdown(
                        f"<div class='text-card'>📝 <b>Printed Text</b>: "
                        f"\"{synthesis['ocr_text']}\"</div>",
                        unsafe_allow_html=True
                    )

                # Audio Playback
                st.markdown("#### 🔊 Spoken Description")
                st.write(f"*{spoken_text}*")
                
                audio_bytes = tts.to_audio_bytes(spoken_text)
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/mp3", autoplay=auto_play_audio)
                
                if play_through_speakers:
                    tts.speak(spoken_text)

                # Show annotated detections
                with st.expander("Show Annotated Computer Vision Feed"):
                    st.image(annotated_img, caption="Detected Objects and Bounding Boxes", use_container_width=True)


# =========================================================================
# TAB 2: VOICE & VISUAL QUESTION ANSWERING
# =========================================================================
with tabs[1]:
    st.markdown("#### Ask questions about what is in view (Speech or Text)")
    
    vqa_source = st.radio(
        "Select Image for Q&A:",
        ["Current Scene (Uploaded/Sample)", "Take New Photo"],
        horizontal=True,
        key="vqa_source"
    )
    
    vqa_image = None
    if vqa_source == "Current Scene (Uploaded/Sample)":
        sample_path = os.path.join(PROJECT_ROOT, "sample_images", "bus.jpg")
        if os.path.exists(sample_path):
            vqa_image = load_image(sample_path)
            st.image(vqa_image, caption="Active Scene", width=400)
    else:
        vqa_cam = st.camera_input("Capture for Q&A", key="vqa_cam")
        if vqa_cam is not None:
            vqa_image = load_image(vqa_cam.read())
            st.image(vqa_image, caption="Captured Scene", width=400)

    if vqa_image is not None:
        st.markdown("---")
        st.markdown("##### 🎙️ Ask via Voice Recording:")
        recorded_audio = None
        # Streamlit >=1.37 has st.audio_input
        if hasattr(st, "audio_input"):
            recorded_audio = st.audio_input("Record your voice question")
        else:
            recorded_audio = st.file_uploader("Upload an audio question (.wav, .mp3)", type=["wav", "mp3"])

        question_text = ""
        if recorded_audio is not None:
            with st.spinner("Transcribing your speech via Whisper..."):
                recognizer = get_speech_recognizer()
                audio_bytes = recorded_audio.read()
                transcribed = recognizer.transcribe(audio_bytes)
                recognizer.unload()
                if transcribed:
                    st.success(f"Recognized Speech: **\"{transcribed}\"**")
                    question_text = transcribed

        st.markdown("##### Or Type Your Question:")
        typed_question = st.text_input(
            "Enter question here:", 
            value=question_text,
            placeholder="e.g. What color is the bus? / How many people are there?"
        )

        if st.button("💬 Get Answer", key="btn_vqa"):
            target_q = typed_question or question_text
            if not target_q.strip():
                st.warning("Please provide a question via voice or text input.")
            else:
                with st.spinner("Analyzing image to answer your question..."):
                    vqa = VQAEngine()
                    answer = vqa.answer_question(vqa_image, target_q)
                    vqa.unload()

                st.markdown(
                    f"<div class='scene-card'><b>Question:</b> {target_q}<br/>"
                    f"<b>Answer:</b> <span style='color: #FFD700; font-size: 1.3rem;'>{answer}</span></div>",
                    unsafe_allow_html=True
                )

                spoken_ans = f"The answer to '{target_q}' is: {answer}."
                audio_bytes = tts.to_audio_bytes(spoken_ans)
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/mp3", autoplay=auto_play_audio)

                if play_through_speakers:
                    tts.speak(spoken_ans)


# =========================================================================
# TAB 3: OBJECT & OBSTACLE DETECTOR ONLY
# =========================================================================
with tabs[2]:
    st.markdown("#### Dedicated Object Detection & Directional Proximity")
    det_img_file = st.file_uploader("Upload image for object detection", type=["jpg", "png"], key="det_tab_upload")
    
    det_target_img = None
    if det_img_file is not None:
        det_target_img = load_image(det_img_file.read())
    else:
        sample_path = os.path.join(PROJECT_ROOT, "sample_images", "bus.jpg")
        if os.path.exists(sample_path):
            det_target_img = load_image(sample_path)

    if det_target_img is not None:
        col_d1, col_d2 = st.columns([1, 1])
        with col_d1:
            st.image(det_target_img, caption="Input Scene", use_container_width=True)
        with col_d2:
            conf = st.slider("Confidence Threshold", 0.15, 0.90, 0.35, 0.05)
            if st.button("Run Object Detection", key="btn_det_tab"):
                with st.spinner("Detecting objects..."):
                    detector = ObjectDetector(model_name="yolov8n.pt", confidence_threshold=conf)
                    detections = detector.detect(det_target_img)
                    annotated = detector.annotate(det_target_img, detections)
                    spoken = detector.generate_spoken_summary(detections)
                    detector.unload()

                st.image(annotated, caption="Annotated Objects", use_container_width=True)
                st.write(f"**Spoken Summary**: {spoken}")
                
                audio_bytes = tts.to_audio_bytes(spoken)
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/mp3", autoplay=auto_play_audio)


# =========================================================================
# TAB 4: OCR DOCUMENT & SIGN READER ONLY
# =========================================================================
with tabs[3]:
    st.markdown("#### Dedicated Sign, Label, and Document Text Reader")
    ocr_img_file = st.file_uploader("Upload image for text extraction", type=["jpg", "png"], key="ocr_tab_upload")

    ocr_target_img = None
    if ocr_img_file is not None:
        ocr_target_img = load_image(ocr_img_file.read())
    else:
        ocr_sample_path = os.path.join(PROJECT_ROOT, "ocr_result.jpg")
        if os.path.exists(ocr_sample_path):
            ocr_target_img = load_image(ocr_sample_path)

    if ocr_target_img is not None:
        col_o1, col_o2 = st.columns([1, 1])
        with col_o1:
            st.image(ocr_target_img, caption="Input Image", use_container_width=True)
        with col_o2:
            if st.button("Read Printed Text", key="btn_ocr_tab"):
                with st.spinner("Extracting printed text..."):
                    reader = OCRReader(confidence_threshold=0.25)
                    results = reader.read_text(ocr_target_img)
                    annotated = reader.annotate(ocr_target_img, results)
                    spoken = reader.generate_spoken_summary(results)
                    reader.unload()

                st.image(annotated, caption="Extracted Text Blocks", use_container_width=True)
                st.write(f"**Extracted Text**: {spoken}")

                audio_bytes = tts.to_audio_bytes(spoken)
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/mp3", autoplay=auto_play_audio)


# =========================================================================
# TAB 5: ABOUT, ARCHITECTURE & LIMITATIONS
# =========================================================================
with tabs[4]:
    st.markdown("### 📚 Academic Project Overview")
    st.markdown("""
    **Project Title**: *AI-Powered Visual Assistant for Visually Impaired People Using Computer Vision and NLP*
    
    #### 🏗️ Architecture & Pretrained Models Used:
    1. **Object Detection**: Pretrained **Ultralytics YOLOv8n** on COCO (80 classes). Computes bounding box centroids for horizontal positioning (*left*, *center*, *right*) and area ratios for proximity alerts.
    2. **Text Extraction (OCR)**: Pretrained **EasyOCR** (CRAFT text detector + ResNet/LSTM recognizer). Clusters text in natural reading order (top-to-bottom, left-to-right).
    3. **Scene Captioning**: Pretrained **Salesforce BLIP Base** (`Salesforce/blip-image-captioning-base`) running in FP16 precision.
    4. **Visual Question Answering**: Pretrained **Salesforce BLIP VQA Base** (`Salesforce/blip-vqa-base`) for zero-shot natural language queries.
    5. **Speech Recognition**: Pretrained **OpenAI Whisper** (`tiny`/`base`) with pure-Python `soundfile` decoding.
    6. **Text-to-Speech**: Dual-engine synthesis using **pyttsx3** (offline Windows SAPI5) and **gTTS** (in-browser MP3 streaming).

    #### ⚠️ Safety & Technical Limitations:
    - **No Calibrated 3D Metric Depth**: Distances are estimated heuristically from 2D bounding box area. The system cannot measure exact metric distances.
    - **Not an Autonomous Mobility Device**: The software must **never** be used for autonomous obstacle avoidance, traffic navigation, or hazard avoidance.
    - **Lighting & Motion Sensitivity**: Heavy camera blur, extreme shadows, or glare will degrade OCR and detection accuracy.
    - **Probabilistic Hallucinations**: Vision-language models may occasionally output inaccurate details when queried about ambiguous visual features.
    """)
