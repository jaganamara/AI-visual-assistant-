"""
Unit tests for SceneCaptioner, VQAEngine, and ContextSynthesizer modules.
"""

import os
import pytest
from PIL import Image

from src.utils.context_synthesizer import ContextSynthesizer
from src.vision.utils import load_image

# Path to local sample bus image
BUS_IMG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sample_images", "bus.jpg")


def test_context_synthesizer_prioritization():
    """Verify that ContextSynthesizer prioritizes immediate hazards over general descriptions."""
    caption = "A street scene with vehicles and pedestrians"
    detections = [
        {"label": "bus", "direction": "directly in front", "urgency": "HIGH", "proximity": "immediate obstacle (very close)"},
        {"label": "person", "direction": "on your left", "urgency": "MEDIUM", "proximity": "nearby"},
    ]
    ocr = [
        {"text": "METRO LINE 4", "direction": "on your right"}
    ]

    result = ContextSynthesizer.synthesize(caption=caption, detections=detections, ocr_results=ocr)

    narrative = result["spoken_narrative"]
    assert "Caution: Immediate obstacle close ahead" in narrative
    assert "bus (directly in front)" in narrative
    assert "Scene view:" in narrative
    assert "Detected text reads: \"METRO LINE 4\"" in narrative

    # Safety check: Hazard alert MUST appear before the general scene view
    hazard_idx = narrative.find("Caution: Immediate obstacle")
    scene_idx = narrative.find("Scene view:")
    assert hazard_idx < scene_idx


def test_scene_captioner_inference():
    """Verify BLIP captioner produces a valid description on a sample image."""
    from src.nlp.scene_captioner import SceneCaptioner

    captioner = SceneCaptioner()
    assert os.path.exists(BUS_IMG_PATH), f"Sample image {BUS_IMG_PATH} missing!"

    img = load_image(BUS_IMG_PATH)
    caption = captioner.generate_caption(img)

    assert isinstance(caption, str)
    assert len(caption) > 5
    print(f"Generated Caption: {caption}")

    # Test unload
    captioner.unload()
    assert captioner.model is None


def test_vqa_engine_inference():
    """Verify BLIP VQA produces an accurate answer to a natural language question."""
    from src.nlp.vqa_engine import VQAEngine

    vqa = VQAEngine()
    assert os.path.exists(BUS_IMG_PATH)

    img = load_image(BUS_IMG_PATH)
    question = "What vehicle is in the picture?"
    answer = vqa.answer_question(img, question)

    assert isinstance(answer, str)
    assert len(answer) > 0
    print(f"VQA Answer to '{question}': {answer}")
    assert "bus" in answer.lower()

    # Test unload
    vqa.unload()
    assert vqa.model is None
