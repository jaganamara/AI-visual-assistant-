"""
Unit tests for ObjectDetector and Spatial Reasoning module.
"""

import pytest
from PIL import Image
import torch

from src.vision.object_detector import ObjectDetector
from src.vision.utils import create_synthetic_test_image


def test_spatial_position_logic():
    """Verify directional categorization and proximity thresholds."""
    img_w, img_h = 1000, 1000

    # Left object
    box_left = [50, 100, 200, 300]
    info_left = ObjectDetector._compute_spatial_position(box_left, img_w, img_h)
    assert info_left["direction"] == "on your left"

    # Center object
    box_center = [450, 200, 550, 400]
    info_center = ObjectDetector._compute_spatial_position(box_center, img_w, img_h)
    assert info_center["direction"] == "directly in front"

    # Right object
    box_right = [800, 200, 950, 400]
    info_right = ObjectDetector._compute_spatial_position(box_right, img_w, img_h)
    assert info_right["direction"] == "on your right"

    # Large object (high proximity / obstacle)
    box_large = [200, 100, 800, 800]  # area = 600 * 700 = 420,000 / 1,000,000 = 42%
    info_large = ObjectDetector._compute_spatial_position(box_large, img_w, img_h)
    assert info_large["urgency"] == "HIGH"
    assert "immediate obstacle" in info_large["proximity"]


def test_detector_inference_on_synthetic_image():
    """Verify detector initialization, inference format, and output structure."""
    detector = ObjectDetector(model_name="yolov8n.pt", confidence_threshold=0.25)
    test_img = create_synthetic_test_image(640, 480)

    detections = detector.detect(test_img)
    assert isinstance(detections, list)

    # If any objects detected, check required fields
    for det in detections:
        assert "label" in det
        assert "confidence" in det
        assert "box" in det
        assert len(det["box"]) == 4
        assert "direction" in det
        assert "proximity" in det
        assert "urgency" in det
        assert "area_percentage" in det

    # Test annotation function
    annotated = detector.annotate(test_img, detections)
    assert isinstance(annotated, Image.Image)
    assert annotated.size == (640, 480)

    # Test spoken summary
    summary = detector.generate_spoken_summary(detections)
    assert isinstance(summary, str)
    assert len(summary) > 0

    # Test unload
    detector.unload()
    assert detector.model is None


def test_detector_cpu_fallback():
    """Verify detector executes safely when forced to CPU device."""
    detector = ObjectDetector(model_name="yolov8n.pt")
    detector.device = torch.device("cpu")
    test_img = create_synthetic_test_image(320, 240)

    detections = detector.detect(test_img)
    assert isinstance(detections, list)
    detector.unload()
