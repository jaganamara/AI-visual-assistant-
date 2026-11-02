"""
Unit tests for OCRReader module and natural reading order sorting.
"""

import pytest
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from src.ocr.ocr_reader import OCRReader


def test_spatial_tagging_for_text():
    """Verify directional categorization for text positions."""
    img_w, img_h = 1000, 1000

    # Left text box
    poly_left = [[50, 100], [200, 100], [200, 150], [50, 150]]
    assert OCRReader._compute_text_spatial_info(poly_left, img_w, img_h) == "on your left"

    # Center text box
    poly_center = [[400, 100], [600, 100], [600, 150], [400, 150]]
    assert OCRReader._compute_text_spatial_info(poly_center, img_w, img_h) == "in front"

    # Right text box
    poly_right = [[800, 100], [950, 100], [950, 150], [800, 150]]
    assert OCRReader._compute_text_spatial_info(poly_right, img_w, img_h) == "on your right"


def test_reading_order_sorting():
    """Verify top-to-bottom and left-to-right reading order sorting."""
    # Simulating 3 text boxes:
    # Item 1: Line 2 (y=100), x=50
    # Item 2: Line 1 (y=20), x=200
    # Item 3: Line 1 (y=22), x=50
    items = [
        {"text": "Line 2", "polygon": [[50, 100], [150, 100], [150, 140], [50, 140]]},
        {"text": "World", "polygon": [[200, 20], [300, 20], [300, 60], [200, 60]]},
        {"text": "Hello", "polygon": [[50, 22], [150, 22], [150, 60], [50, 60]]},
    ]

    sorted_items = OCRReader._sort_reading_order(items, line_threshold=15.0)
    texts = [it["text"] for it in sorted_items]

    # Hello and World should be line 1 (Hello before World), followed by Line 2
    assert texts == ["Hello", "World", "Line 2"]


def test_ocr_inference_on_synthetic_sign():
    """Create a high-contrast test image with clear text and run OCR."""
    # Render synthetic sign: White background, black bold text
    img = Image.new("RGB", (600, 300), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Big text
    draw.text((50, 60), "ROOM 404", fill=(0, 0, 0))
    draw.text((50, 160), "EMERGENCY EXIT", fill=(0, 0, 0))

    reader = OCRReader(confidence_threshold=0.20)
    results = reader.read_text(img)

    assert isinstance(results, list)

    for item in results:
        assert "text" in item
        assert "confidence" in item
        assert "polygon" in item
        assert "box" in item
        assert "direction" in item

    # Test annotation
    annotated = reader.annotate(img, results)
    assert isinstance(annotated, Image.Image)
    assert annotated.size == (600, 300)

    # Test spoken summary
    spoken = reader.generate_spoken_summary(results)
    assert isinstance(spoken, str)
    assert len(spoken) > 0

    # Test unload
    reader.unload()
    assert reader.reader is None
