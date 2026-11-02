"""
AI Visual Assistant - Production UI Redesign.
Multimodal assistive vision web application with YOLOv8 object detection,
Salesforce BLIP scene captioning & VQA, EasyOCR text extraction,
and Whisper / gTTS speech interaction.
Recreated to pixel-level fidelity matching the reference design specification.
"""

import os
import sys
import time
from typing import Optional, List, Dict, Any
from PIL import Image
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st

# Hardware and AI modules
from src.utils.device_manager import DeviceManager, get_cache_dir
from src.vision.object_detector import ObjectDetector
from src.ocr.ocr_reader import OCRReader
from src.nlp.scene_captioner import SceneCaptioner
from src.nlp.vqa_engine import VQAEngine
from src.speech.speech_recognizer import SpeechRecognizer
from src.speech.text_to_speech import TextToSpeech
from src.utils.context_synthesizer import ContextSynthesizer
from src.vision.utils import load_image

# UI modules
from src.ui.styles import get_global_css
from src.ui.components import (
    render_app_header,
    render_workspace_hero,
    render_getting_started_card,
    draw_styled_detections,
    render_audio_waveform,
    get_eye_logo_svg,
)
from src.ui.history import ActivityHistory


# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Visual Assistant",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State
if "active_nav" not in st.session_state:
    st.session_state.active_nav = "Scene Analysis"

# Read query params navigation if present
if "nav" in st.query_params:
    target_nav = st.query_params["nav"]
    valid_navs = ["Scene Analysis", "Voice & Visual Q&A", "Object Detection", "Text Reader (OCR)", "Activity History", "Settings"]
    if target_nav in valid_navs:
        st.session_state.active_nav = target_nav

if "font_scale" not in st.session_state:
    st.session_state.font_scale = 1.0

if "high_contrast" not in st.session_state:
    st.session_state.high_contrast = False

if "play_through_speakers" not in st.session_state:
    st.session_state.play_through_speakers = False

if "auto_play_audio" not in st.session_state:
    st.session_state.auto_play_audio = True

if "current_image" not in st.session_state:
    st.session_state.current_image = None

if "current_annotated_img" not in st.session_state:
    st.session_state.current_annotated_img = None

if "current_detections" not in st.session_state:
    st.session_state.current_detections = []

if "current_caption" not in st.session_state:
    st.session_state.current_caption = ""

if "current_ocr_results" not in st.session_state:
    st.session_state.current_ocr_results = []

if "spoken_narrative" not in st.session_state:
    st.session_state.spoken_narrative = ""

if "is_speaking" not in st.session_state:
    st.session_state.is_speaking = False

if "audio_bytes" not in st.session_state:
    st.session_state.audio_bytes = None

if "input_mode" not in st.session_state:
    st.session_state.input_mode = "none"

if "vqa_last_q" not in st.session_state:
    st.session_state.vqa_last_q = ""

if "vqa_last_a" not in st.session_state:
    st.session_state.vqa_last_a = ""

if "getting_started_step" not in st.session_state:
    st.session_state.getting_started_step = 1

# Inject scoped high-fidelity stylesheet
st.markdown(
    get_global_css(
        font_scale=st.session_state.font_scale,
        high_contrast=st.session_state.high_contrast
    ),
    unsafe_allow_html=True
)


# --- CACHED SINGLETON SERVICE INITIALIZERS ---
@st.cache_resource
def get_tts() -> TextToSpeech:
    return TextToSpeech()

@st.cache_resource
def get_speech_recognizer() -> SpeechRecognizer:
    return SpeechRecognizer(model_size="tiny")


def run_full_scene_pipeline(image: Image.Image, record_task: str = "Scene Analysis") -> None:
    """Executes the full AI pipeline on the target image and caches outputs."""
    with st.spinner("Analyzing surroundings with local vision & language models..."):
        # 1. Object Detection (YOLOv8n)
        detector = ObjectDetector(model_name="yolov8n.pt", confidence_threshold=0.35)
        detections = detector.detect(image)
        annotated_img = draw_styled_detections(image, detections)
        detector.unload()

        # 2. Text Extraction (EasyOCR)
        ocr = OCRReader(confidence_threshold=0.25)
        ocr_results = ocr.read_text(image)
        ocr.unload()

        # 3. Scene Captioning (BLIP)
        captioner = SceneCaptioner()
        caption = captioner.generate_caption(image)
        captioner.unload()

        # 4. Context Synthesis
        synthesis = ContextSynthesizer.synthesize(
            caption=caption,
            detections=detections,
            ocr_results=ocr_results
        )
        spoken_text = synthesis["spoken_narrative"]

        # 5. Text-to-Speech audio generation
        tts = get_tts()
        audio_data = tts.to_audio_bytes(spoken_text)

        # Update Session State
        st.session_state.current_image = image
        st.session_state.current_detections = detections
        st.session_state.current_annotated_img = annotated_img
        st.session_state.current_caption = caption
        st.session_state.current_ocr_results = ocr_results
        st.session_state.spoken_narrative = spoken_text
        st.session_state.audio_bytes = audio_data

        # Record in activity history
        ActivityHistory.add_record(
            task_type=record_task,
            summary=f"Detected {len(detections)} objects. {caption.capitalize() if caption else ''}",
            details={
                "objects_count": len(detections),
                "caption": caption,
                "ocr_count": len(ocr_results)
            },
            image_preview=annotated_img,
            spoken_text=spoken_text
        )


# --- AUTO-LOAD DEFAULT SAMPLE SCENE IF FRESH START ---
if st.session_state.current_image is None:
    # Check for city sidewalk sample first, or fallback to bus.jpg
    sample_sidewalk = os.path.join(PROJECT_ROOT, "sample_images", "city_sidewalk.jpg")
    sample_bus = os.path.join(PROJECT_ROOT, "sample_images", "bus.jpg")
    initial_sample = sample_sidewalk if os.path.exists(sample_sidewalk) else (sample_bus if os.path.exists(sample_bus) else None)
    
    if initial_sample:
        try:
            loaded_img = load_image(initial_sample)
            run_full_scene_pipeline(loaded_img, record_task="Initial Scene Analysis")
        except Exception as e:
            # Fallback gracefully if model initialization is pending
            st.session_state.current_image = load_image(initial_sample)


# --- APPLICATION HEADER ---
vram_info = DeviceManager.get_vram_info()
render_app_header(gpu_info=vram_info, user_initials="AJ")


# --- LEFT SIDEBAR NAVIGATION ---
with st.sidebar:
    st.markdown('<div class="sidebar-nav-container">', unsafe_allow_html=True)
    
    nav_items = [
        ("Scene Analysis", "🏠  Scene Analysis"),
        ("Voice & Visual Q&A", "🎙️  Voice & Visual Q&A"),
        ("Object Detection", "🎯  Object Detection"),
        ("Text Reader (OCR)", "📄  Text Reader (OCR)"),
        ("Activity History", "⏱️  Activity History"),
        ("Settings", "⚙️  Settings"),
    ]

    for nav_id, nav_label in nav_items:
        is_active = (st.session_state.active_nav == nav_id)
        marker = '<div class="nav-active-trigger"></div>' if is_active else '<div class="nav-inactive-trigger"></div>'
        st.markdown(marker, unsafe_allow_html=True)
        if st.button(nav_label, key=f"nav_btn_{nav_id}", use_container_width=True):
            st.session_state.active_nav = nav_id
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    # Getting Started Card at bottom
    render_getting_started_card()
    
    col_step1, col_step2 = st.columns([4, 1])
    with col_step2:
        st.markdown('<div class="btn-stepper-trigger"></div>', unsafe_allow_html=True)
        if st.button("→", key="btn_next_guide_step", help="Next quick guide tip"):
            st.session_state.getting_started_step = (st.session_state.getting_started_step % 3) + 1
            st.rerun()


# =========================================================================
# VIEW 1: SCENE ANALYSIS (PRIMARY DASHBOARD MATCHING REFERENCE IMAGE)
# =========================================================================
if st.session_state.active_nav == "Scene Analysis":
    render_workspace_hero()

    col_left, col_right = st.columns([1.02, 1.05], gap="large")

    # --- LEFT CARD: VISUAL PREVIEW & CONTROLS ---
    with col_left:
        st.markdown('<div class="dash-card">', unsafe_allow_html=True)

        # Image Preview Area
        if st.session_state.current_annotated_img is not None:
            st.markdown("""
            <div class="preview-box">
                <div class="preview-expand-btn" title="Expand View">⛶</div>
            </div>
            """, unsafe_allow_html=True)
            st.image(st.session_state.current_annotated_img, use_container_width=True)
        elif st.session_state.current_image is not None:
            st.image(st.session_state.current_image, use_container_width=True)
        else:
            st.markdown("""
            <div class="preview-box">
                <div class="preview-empty">
                    <div class="preview-empty-icon">👁️</div>
                    <div style="font-weight: 600; font-size: 1.05rem; color: #F4D9A0; margin-bottom: 4px;">No Visual Scene Loaded</div>
                    <div style="font-size: 0.85rem;">Capture live camera feed, upload a photo, or load a sample scene below.</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Three Action Buttons Row
        col_b1, col_b2, col_b3 = st.columns(3, gap="small")
        with col_b1:
            st.markdown('<div class="btn-gold-trigger"></div>', unsafe_allow_html=True)
            if st.button("📷 Open Camera", key="btn_open_cam", use_container_width=True):
                st.session_state.input_mode = "camera" if st.session_state.input_mode != "camera" else "none"
                st.rerun()

        with col_b2:
            st.markdown('<div class="btn-mocha-trigger"></div>', unsafe_allow_html=True)
            if st.button("📤 Upload Image", key="btn_upload_img", use_container_width=True):
                st.session_state.input_mode = "upload" if st.session_state.input_mode != "upload" else "none"
                st.rerun()

        with col_b3:
            st.markdown('<div class="btn-mocha-trigger"></div>', unsafe_allow_html=True)
            if st.button("🖼️ Use Sample Scene", key="btn_sample_scene", use_container_width=True):
                st.session_state.input_mode = "none"
                sample_p = os.path.join(PROJECT_ROOT, "sample_images", "city_sidewalk.jpg")
                if not os.path.exists(sample_p):
                    sample_p = os.path.join(PROJECT_ROOT, "sample_images", "bus.jpg")
                if os.path.exists(sample_p):
                    img = load_image(sample_p)
                    run_full_scene_pipeline(img, record_task="Sample Scene Analysis")
                    st.rerun()

        # Interactive Inputs when toggled
        if st.session_state.input_mode == "camera":
            st.markdown("---")
            cam_shot = st.camera_input("Capture current viewpoint", key="cam_widget")
            if cam_shot is not None:
                img = load_image(cam_shot.read())
                run_full_scene_pipeline(img, record_task="Webcam Live Analysis")
                st.session_state.input_mode = "none"
                st.rerun()

        elif st.session_state.input_mode == "upload":
            st.markdown("---")
            up_file = st.file_uploader("Select JPG or PNG image", type=["jpg", "jpeg", "png"], key="up_widget")
            if up_file is not None:
                img = load_image(up_file.read())
                run_full_scene_pipeline(img, record_task="Uploaded Image Analysis")
                st.session_state.input_mode = "none"
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    # --- RIGHT CARD: ANALYSIS RESULTS ---
    with col_right:
        st.markdown('<div class="dash-card">', unsafe_allow_html=True)

        # Results Header with Sparkle and Completion Badge
        is_complete = bool(st.session_state.current_image is not None)
        status_pill_html = (
            '<div class="status-badge-complete"><span class="status-dot-green"></span> Analysis Complete</div>'
            if is_complete else
            '<div class="status-badge-ready">Ready to Analyze</div>'
        )

        st.markdown(f"""
        <div class="results-header">
            <div class="results-title-group">
                <span class="sparkle-icon">✨</span>
                <span class="results-title">Analysis Results</span>
            </div>
            {status_pill_html}
        </div>
        """, unsafe_allow_html=True)

        # Section A: Detected Objects
        det_count = len(st.session_state.current_detections)
        with st.expander(f"📦 Detected Objects  ({det_count})", expanded=True):
            if st.session_state.current_detections:
                pills_html = []
                icon_map = {
                    "person": "👤",
                    "bicycle": "🚲",
                    "car": "🚗",
                    "bus": "🚌",
                    "traffic light": "🚦",
                    "traffic sign": "⚠️",
                    "backpack": "🎒",
                    "handbag": "👜",
                    "dog": "🐕"
                }
                for d in st.session_state.current_detections:
                    lbl = d.get("label", "object")
                    icon = icon_map.get(lbl.lower(), "🏷️")
                    conf = d.get("confidence", 0.0)
                    urgency_cls = "object-pill-hazard" if d.get("urgency") == "HIGH" else "object-pill"
                    pills_html.append(
                        f'<div class="{urgency_cls}">'
                        f'<span class="object-pill-icon">{icon}</span>'
                        f'<span>{lbl.title()} ({conf:.2f})</span>'
                        f'</div>'
                    )
                st.markdown(f'<div class="object-pills-container">{"".join(pills_html)}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<span style="color: #A89887; font-style: italic;">No prominent objects detected in view.</span>', unsafe_allow_html=True)

        # Section B: Scene Summary
        with st.expander("📄 Scene Summary", expanded=True):
            if st.session_state.current_caption:
                clean_cap = st.session_state.current_caption.strip()
                clean_cap = clean_cap[0].upper() + clean_cap[1:]
                if not clean_cap.endswith("."):
                    clean_cap += "."
                st.markdown(f'<div style="color: #E2D3C4; line-height: 1.6;">{clean_cap}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<span style="color: #A89887; font-style: italic;">Load or capture a scene to generate a natural language overview.</span>', unsafe_allow_html=True)

        # Section C: Read Text (OCR)
        with st.expander("🔤 Read Text (OCR)", expanded=True):
            if st.session_state.current_ocr_results:
                texts = [r["text"] for r in st.session_state.current_ocr_results if len(r.get("text", "").strip()) > 1]
                if texts:
                    st.markdown(f'<div style="color: #F4D9A0; font-weight: 500;">"{ " ".join(texts) }"</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div style="color: #A89887;">No readable text detected in this scene.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div style="color: #A89887;">No readable text detected in this scene.</div>', unsafe_allow_html=True)

        # Bottom Voice Output Row
        st.markdown('<div class="voice-output-row">', unsafe_allow_html=True)
        col_v1, col_v2 = st.columns([1.1, 1.6], gap="medium")
        
        with col_v1:
            st.markdown('<div class="btn-speak-trigger"></div>', unsafe_allow_html=True)
            if st.button("🔊  Speak Results", key="btn_speak_dashboard", use_container_width=True):
                if st.session_state.spoken_narrative:
                    tts = get_tts()
                    st.session_state.is_speaking = True
                    audio_data = tts.to_audio_bytes(st.session_state.spoken_narrative)
                    st.session_state.audio_bytes = audio_data
                    if st.session_state.play_through_speakers:
                        tts.speak(st.session_state.spoken_narrative)
                else:
                    st.info("No scene narrative generated yet.")

        with col_v2:
            st.markdown(render_audio_waveform(is_speaking=st.session_state.is_speaking), unsafe_allow_html=True)

        # Hidden or auto-play audio player
        if st.session_state.audio_bytes is not None:
            st.audio(
                st.session_state.audio_bytes,
                format="audio/mp3",
                autoplay=st.session_state.auto_play_audio
            )

        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- FULL-WIDTH BOTTOM QUESTION BAR ---
    st.markdown('<div class="question-bar-card">', unsafe_allow_html=True)
    
    col_q1, col_q2, col_q3 = st.columns([1.1, 3.2, 0.45], gap="medium")
    
    with col_q1:
        st.markdown("""
        <div class="question-label">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#F4D9A0" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path>
                <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
                <line x1="12" y1="19" x2="12" y2="23"></line>
                <line x1="8" y1="23" x2="16" y2="23"></line>
            </svg>
            <span>Ask a Question about this Scene</span>
        </div>
        """, unsafe_allow_html=True)

    with col_q2:
        typed_q = st.text_input(
            "Query Scene",
            placeholder="e.g., What is in front of me?",
            label_visibility="collapsed",
            key="dashboard_question_input"
        )

    with col_q3:
        st.markdown('<div class="btn-send-trigger"></div>', unsafe_allow_html=True)
        send_clicked = st.button("✈", key="btn_submit_vqa", help="Ask Visual Question")

    # Handle Question submission
    if send_clicked and typed_q.strip():
        if st.session_state.current_image is None:
            st.warning("Please load or capture an image first.")
        else:
            with st.spinner("Analyzing scene with BLIP VQA..."):
                vqa = VQAEngine()
                answer = vqa.answer_question(st.session_state.current_image, typed_q.strip())
                vqa.unload()

                st.session_state.vqa_last_q = typed_q.strip()
                st.session_state.vqa_last_a = answer

                # Spoken audio answer
                spoken_ans = f"Regarding '{typed_q.strip()}': {answer}."
                tts = get_tts()
                ans_audio = tts.to_audio_bytes(spoken_ans)
                st.session_state.audio_bytes = ans_audio
                if st.session_state.play_through_speakers:
                    tts.speak(spoken_ans)

                # Record in activity history
                ActivityHistory.add_record(
                    task_type="Visual Q&A",
                    summary=f"Q: '{typed_q.strip()}' -> A: {answer}",
                    details={"question": typed_q.strip(), "answer": answer},
                    spoken_text=spoken_ans
                )

    # Show active Q&A response card if answered
    if st.session_state.vqa_last_q and st.session_state.vqa_last_a:
        st.markdown(f"""
        <div class="vqa-answer-card">
            <div class="vqa-q-text">Q: {st.session_state.vqa_last_q}</div>
            <div class="vqa-a-text">A: <span style="color: #F4D9A0;">{st.session_state.vqa_last_a}</span></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# =========================================================================
# VIEW 2: VOICE & VISUAL Q&A
# =========================================================================
elif st.session_state.active_nav == "Voice & Visual Q&A":
    st.markdown('<div class="workspace-title">Voice & Visual Question Answering</div>', unsafe_allow_html=True)
    st.markdown('<div class="workspace-subtitle">Ask questions aloud using your microphone or type specific queries about what is in view.</div>', unsafe_allow_html=True)
    st.markdown("<br/>", unsafe_allow_html=True)

    col_vqa1, col_vqa2 = st.columns([1, 1], gap="large")

    with col_vqa1:
        st.markdown('<div class="dash-card">', unsafe_allow_html=True)
        st.markdown('<div style="font-weight: 600; color: #F4D9A0; margin-bottom: 10px;">Active Scene Reference</div>', unsafe_allow_html=True)
        if st.session_state.current_image is not None:
            st.image(st.session_state.current_image, use_container_width=True)
        else:
            st.info("No active scene. Please load a sample or upload an image in Scene Analysis.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_vqa2:
        st.markdown('<div class="dash-card">', unsafe_allow_html=True)
        st.markdown('<div style="font-weight: 600; color: #F4D9A0; margin-bottom: 10px;">🎙️ Voice Input (OpenAI Whisper)</div>', unsafe_allow_html=True)

        voice_query = None
        if hasattr(st, "audio_input"):
            rec_audio = st.audio_input("Record voice query", key="voice_q_rec")
            if rec_audio is not None:
                with st.spinner("Transcribing speech with Whisper..."):
                    recognizer = get_speech_recognizer()
                    voice_query = recognizer.transcribe(rec_audio.read())
                    recognizer.unload()
                    if voice_query:
                        st.success(f"Recognized: **\"{voice_query}\"**")

        typed_q = st.text_input("Or enter your question:", value=voice_query or "", placeholder="e.g. What color is the bicycle?", key="typed_vqa_tab")

        st.markdown('<div class="btn-speak-trigger"></div>', unsafe_allow_html=True)
        if st.button("💬 Get Spoken Answer", key="btn_vqa_tab_exec", use_container_width=True):
            query = typed_q or voice_query
            if not query:
                st.warning("Please record a voice question or type one.")
            elif st.session_state.current_image is None:
                st.warning("Please load an image reference first.")
            else:
                with st.spinner("Analyzing image..."):
                    vqa = VQAEngine()
                    ans = vqa.answer_question(st.session_state.current_image, query)
                    vqa.unload()

                    st.markdown(f"""
                    <div class="vqa-answer-card">
                        <div class="vqa-q-text">Question: {query}</div>
                        <div class="vqa-a-text">Answer: <span style="color: #F4D9A0; font-size: 1.25rem;">{ans}</span></div>
                    </div>
                    """, unsafe_allow_html=True)

                    spoken = f"The answer to '{query}' is: {ans}."
                    tts = get_tts()
                    aud = tts.to_audio_bytes(spoken)
                    if aud:
                        st.audio(aud, format="audio/mp3", autoplay=True)
                    if st.session_state.play_through_speakers:
                        tts.speak(spoken)

                    ActivityHistory.add_record(
                        task_type="Visual Q&A",
                        summary=f"Q: '{query}' -> A: {ans}",
                        details={"question": query, "answer": ans},
                        spoken_text=spoken
                    )

        st.markdown('</div>', unsafe_allow_html=True)


# =========================================================================
# VIEW 3: OBJECT DETECTION
# =========================================================================
elif st.session_state.active_nav == "Object Detection":
    st.markdown('<div class="workspace-title">Dedicated Object & Obstacle Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="workspace-subtitle">Spatial localization (left, center, right), proximity hazard estimation, and directional audio guidance.</div>', unsafe_allow_html=True)
    st.markdown("<br/>", unsafe_allow_html=True)

    col_od1, col_od2 = st.columns([1, 1], gap="large")

    with col_od1:
        st.markdown('<div class="dash-card">', unsafe_allow_html=True)
        if st.session_state.current_annotated_img is not None:
            st.image(st.session_state.current_annotated_img, use_container_width=True)
        elif st.session_state.current_image is not None:
            st.image(st.session_state.current_image, use_container_width=True)
        else:
            st.info("No active image loaded. Upload or select an image.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_od2:
        st.markdown('<div class="dash-card">', unsafe_allow_html=True)
        conf = st.slider("Confidence Threshold", 0.15, 0.90, 0.35, 0.05, key="det_slider")
        
        st.markdown('<div class="btn-gold-trigger"></div>', unsafe_allow_html=True)
        if st.button("Run Spatial Object Detection", key="btn_run_det_only", use_container_width=True):
            if st.session_state.current_image is not None:
                with st.spinner("Running YOLOv8 spatial localization..."):
                    detector = ObjectDetector(model_name="yolov8n.pt", confidence_threshold=conf)
                    dets = detector.detect(st.session_state.current_image)
                    ann = draw_styled_detections(st.session_state.current_image, dets)
                    summary = detector.generate_spoken_summary(dets)
                    detector.unload()

                    st.session_state.current_detections = dets
                    st.session_state.current_annotated_img = ann
                    st.session_state.spoken_narrative = summary

                    st.success(f"Detected {len(dets)} objects.")
                    st.write(f"**Spoken Summary**: {summary}")

                    tts = get_tts()
                    aud = tts.to_audio_bytes(summary)
                    if aud:
                        st.audio(aud, format="audio/mp3", autoplay=True)
                    if st.session_state.play_through_speakers:
                        tts.speak(summary)
            else:
                st.warning("Please load a scene image first.")
        st.markdown('</div>', unsafe_allow_html=True)


# =========================================================================
# VIEW 4: TEXT READER (OCR)
# =========================================================================
elif st.session_state.active_nav == "Text Reader (OCR)":
    st.markdown('<div class="workspace-title">Printed Sign & Document Reader (OCR)</div>', unsafe_allow_html=True)
    st.markdown('<div class="workspace-subtitle">Optical character recognition with natural reading order clustering and text-to-speech.</div>', unsafe_allow_html=True)
    st.markdown("<br/>", unsafe_allow_html=True)

    col_ocr1, col_ocr2 = st.columns([1, 1], gap="large")

    with col_ocr1:
        st.markdown('<div class="dash-card">', unsafe_allow_html=True)
        if st.session_state.current_image is not None:
            st.image(st.session_state.current_image, use_container_width=True)
        else:
            st.info("No active image loaded. Upload or select an image.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_ocr2:
        st.markdown('<div class="dash-card">', unsafe_allow_html=True)
        st.markdown('<div class="btn-gold-trigger"></div>', unsafe_allow_html=True)
        if st.button("Extract and Read Text", key="btn_run_ocr_only", use_container_width=True):
            if st.session_state.current_image is not None:
                with st.spinner("Extracting printed text with EasyOCR..."):
                    reader = OCRReader(confidence_threshold=0.25)
                    results = reader.read_text(st.session_state.current_image)
                    ann_ocr = reader.annotate(st.session_state.current_image, results)
                    summary = reader.generate_spoken_summary(results)
                    reader.unload()

                    st.session_state.current_ocr_results = results
                    st.image(ann_ocr, caption="Detected Text Regions", use_container_width=True)
                    st.write(f"**Extracted Text**: {summary}")

                    tts = get_tts()
                    aud = tts.to_audio_bytes(summary)
                    if aud:
                        st.audio(aud, format="audio/mp3", autoplay=True)
                    if st.session_state.play_through_speakers:
                        tts.speak(summary)
            else:
                st.warning("Please load a scene image first.")
        st.markdown('</div>', unsafe_allow_html=True)


# =========================================================================
# VIEW 5: ACTIVITY HISTORY
# =========================================================================
elif st.session_state.active_nav == "Activity History":
    st.markdown('<div class="workspace-title">Activity History</div>', unsafe_allow_html=True)
    st.markdown('<div class="workspace-subtitle">Review recent scene analyses, object detections, questions answered, and audio playback.</div>', unsafe_allow_html=True)
    st.markdown("<br/>", unsafe_allow_html=True)

    records = ActivityHistory.get_records()

    col_h_act, _ = st.columns([1, 4])
    with col_h_act:
        if st.button("🧹 Clear History", key="btn_clear_hist"):
            ActivityHistory.clear()
            st.rerun()

    if not records:
        st.markdown("""
        <div class="dash-card" style="text-align: center; padding: 40px;">
            <div style="font-size: 2.5rem; color: #F4D9A0; margin-bottom: 8px;">⏱️</div>
            <div style="font-size: 1.1rem; color: #FFFFFF; font-weight: 600;">No Activity Records Yet</div>
            <div style="font-size: 0.88rem; color: #A89887; margin-top: 4px;">Recent scene descriptions and visual question answers will be preserved here during your session.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for r in records:
            st.markdown(f"""
            <div class="dash-card" style="margin-bottom: 12px; padding: 16px 20px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="color: #F4D9A0; font-weight: 600; font-size: 0.95rem;">{r['task_type']}</span>
                    <span style="color: #A89887; font-size: 0.8rem;">{r['timestamp']}</span>
                </div>
                <div style="color: #E2D3C4; font-size: 0.95rem; margin-bottom: 8px;">{r['summary']}</div>
            </div>
            """, unsafe_allow_html=True)


# =========================================================================
# VIEW 6: SETTINGS & ACADEMIC LIMITATIONS (REQUIREMENT 15)
# =========================================================================
elif st.session_state.active_nav == "Settings":
    st.markdown('<div class="workspace-title">System Settings & Device Monitor</div>', unsafe_allow_html=True)
    st.markdown('<div class="workspace-subtitle">Manage audio playback preferences, local GPU hardware diagnostics, and accessibility options.</div>', unsafe_allow_html=True)
    st.markdown("<br/>", unsafe_allow_html=True)

    col_s1, col_s2 = st.columns([1, 1], gap="large")

    with col_s1:
        st.markdown('<div class="dash-card">', unsafe_allow_html=True)
        st.markdown('<div style="font-weight: 600; font-size: 1.1rem; color: #F4D9A0; margin-bottom: 14px;">🔊 Audio & Speech Preferences</div>', unsafe_allow_html=True)

        spk = st.checkbox(
            "Play through laptop speakers (offline pyttsx3)",
            value=st.session_state.play_through_speakers,
            help="Speaks aloud using Windows SAPI5 built-in speech synthesis"
        )
        st.session_state.play_through_speakers = spk

        auto_a = st.checkbox("Auto-play browser streaming audio (gTTS)", value=st.session_state.auto_play_audio)
        st.session_state.auto_play_audio = auto_a

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div style="font-weight: 600; font-size: 1.1rem; color: #F4D9A0; margin-bottom: 14px;">♿ Accessibility Display Options</div>', unsafe_allow_html=True)

        f_scale = st.slider("Font Scaling", 0.9, 1.4, float(st.session_state.font_scale), 0.05)
        if f_scale != st.session_state.font_scale:
            st.session_state.font_scale = f_scale
            st.rerun()

        hc = st.checkbox("High Contrast Borders", value=st.session_state.high_contrast)
        if hc != st.session_state.high_contrast:
            st.session_state.high_contrast = hc
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    with col_s2:
        st.markdown('<div class="dash-card">', unsafe_allow_html=True)
        st.markdown('<div style="font-weight: 600; font-size: 1.1rem; color: #F4D9A0; margin-bottom: 14px;">🎮 GPU & Hardware Diagnostics</div>', unsafe_allow_html=True)

        vram = DeviceManager.get_vram_info()
        if vram["cuda_available"]:
            st.success(f"NVIDIA GPU Active: **{vram['device_name']}**")
            col_m1, col_m2 = st.columns(2)
            col_m1.metric("Allocated VRAM", f"{vram['allocated_mb']} MB")
            col_m2.metric("Free VRAM", f"{vram['free_vram_mb']} MB")
        else:
            st.info("Running on CPU Mode")

        st.caption(f"📁 **Models Cache**: `{get_cache_dir()}`")

        if st.button("🧹 Clear GPU VRAM Cache", help="Evicts unused model allocations from RTX memory"):
            DeviceManager.clear_vram_cache()
            st.success("VRAM cache evicted!")
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    # Academic Prototype Notice & Limitations (Relocated per Requirement 15)
    st.markdown('<div class="dash-card" style="margin-top: 1.5rem; border-left: 4px solid #C8792C;">', unsafe_allow_html=True)
    st.markdown('<div style="font-weight: 600; font-size: 1.15rem; color: #F4D9A0; margin-bottom: 10px;">⚠️ Academic Prototype Notice & Technical Limitations</div>', unsafe_allow_html=True)
    st.markdown("""
    This application is strictly an **academic research prototype and assistive cognitive aid**.
    
    - **No Calibrated 3D Metric Depth**: Distances are estimated heuristically from 2D bounding-box area percentage. The system **cannot** measure physical metric distance and must **never** be used as an autonomous navigation system or for life-critical obstacle avoidance.
    - **Environmental Sensitivity**: Performance may degrade under severe lighting, darkness, fast camera motion blur, or heavy lens flare.
    - **Vision-Language Hallucinations**: Vision-language models (BLIP) can occasionally hallucinate plausible-sounding details for ambiguous visual stimuli.
    - **Local Processing**: Model weights and inferences run locally on the detected NVIDIA GeForce RTX 2050 GPU for privacy and zero data leakage.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
