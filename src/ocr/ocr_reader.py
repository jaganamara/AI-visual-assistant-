"""
OCR Module using EasyOCR (CRAFT detector + ResNet/LSTM recognizer).
Provides text detection and recognition for signs, labels, and printed documents,
with natural reading order sorting, spatial tagging, noise filtering, and VRAM management.
"""

import os
import logging
from typing import List, Dict, Any, Union, Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw
import torch

from src.utils.device_manager import DeviceManager, get_cache_dir
from src.vision.utils import load_image

logger = logging.getLogger(__name__)


class OCRReader:
    """
    OCR Engine using EasyOCR with reading order sorting, spatial tagging,
    and memory-safe lifecycle management.
    """

    def __init__(self, languages: Optional[List[str]] = None,
                 confidence_threshold: float = 0.30):
        """
        Initializes the OCR Reader.
        Model weights are stored in models_cache/easyocr/ to keep storage organized.
        """
        self.languages = languages or ["en"]
        self.confidence_threshold = confidence_threshold
        self.cache_dir = get_cache_dir("easyocr")
        self.device = DeviceManager.get_preferred_device()
        self.use_gpu = self.device.type == "cuda"
        self.reader = None

    def _ensure_reader_loaded(self) -> None:
        """Lazy loader: instantiates the EasyOCR Reader only when first needed."""
        if self.reader is None:
            from src.ocr.download_models import ensure_easyocr_models
            ensure_easyocr_models()

            logger.info(f"Loading EasyOCR reader (languages={self.languages}, gpu={self.use_gpu})...")
            try:
                import easyocr
                self.reader = easyocr.Reader(
                    self.languages,
                    gpu=self.use_gpu,
                    model_storage_directory=self.cache_dir,
                    download_enabled=False,
                    verbose=False,
                )
            except (torch.cuda.OutOfMemoryError, RuntimeError, Exception) as e:
                logger.warning(f"Failed to initialize EasyOCR with GPU ({e}). Falling back to CPU...")
                DeviceManager.clear_vram_cache()
                self.use_gpu = False
                import easyocr
                self.reader = easyocr.Reader(
                    self.languages,
                    gpu=False,
                    model_storage_directory=self.cache_dir,
                    download_enabled=False,
                    verbose=False,
                )

    def unload(self) -> None:
        """Explicitly unloads reader and frees VRAM allocations."""
        if self.reader is not None:
            logger.info("Unloading EasyOCR from memory...")
            del self.reader
            self.reader = None
            DeviceManager.clear_vram_cache()

    @staticmethod
    def _compute_text_spatial_info(box_points: List[List[float]], img_w: int, img_h: int) -> str:
        """
        Computes directional location for a text block (left, center, right).
        box_points format: [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        """
        xs = [pt[0] for pt in box_points]
        cx = sum(xs) / (len(xs) * img_w)

        if cx < 0.35:
            return "on your left"
        elif cx > 0.65:
            return "on your right"
        return "in front"

    @staticmethod
    def _sort_reading_order(text_items: List[Dict[str, Any]], line_threshold: float = 20.0) -> List[Dict[str, Any]]:
        """
        Sorts text blocks in natural reading order: top-to-bottom, left-to-right.
        Groups items on roughly the same horizontal line together.
        """
        if not text_items:
            return []

        def get_top_left(item):
            # item["polygon"] is [[x1, y1], [x2, y2], ...]
            pts = item["polygon"]
            min_x = min(p[0] for p in pts)
            min_y = min(p[1] for p in pts)
            return min_y, min_x

        # Initial sort by vertical position
        sorted_by_y = sorted(text_items, key=lambda x: get_top_left(x)[0])

        # Cluster into lines
        lines = []
        current_line = [sorted_by_y[0]]

        for item in sorted_by_y[1:]:
            prev_y = get_top_left(current_line[-1])[0]
            curr_y = get_top_left(item)[0]

            if abs(curr_y - prev_y) <= line_threshold:
                current_line.append(item)
            else:
                # Sort line left-to-right
                current_line.sort(key=lambda x: get_top_left(x)[1])
                lines.extend(current_line)
                current_line = [item]

        if current_line:
            current_line.sort(key=lambda x: get_top_left(x)[1])
            lines.extend(current_line)

        return lines

    def read_text(self, image_input: Union[str, bytes, np.ndarray, Image.Image]) -> List[Dict[str, Any]]:
        """
        Performs OCR on an image and returns structured, filtered text blocks with spatial context.
        """
        pil_img = load_image(image_input)
        img_np = np.array(pil_img)
        img_w, img_h = pil_img.size

        self._ensure_reader_loaded()

        try:
            raw_results = self.reader.readtext(img_np)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            logger.warning(f"EasyOCR CUDA error ({e}). Falling back to CPU...")
            self.unload()
            self.use_gpu = False
            self._ensure_reader_loaded()
            raw_results = self.reader.readtext(img_np)

        extracted_items = []
        for bbox, text, conf in raw_results:
            cleaned_text = text.strip()
            # Filter low confidence or single non-alphanumeric noise characters
            if conf < self.confidence_threshold or (len(cleaned_text) <= 1 and not cleaned_text.isalnum()):
                continue

            # Convert bbox points to native float
            poly_points = [[float(pt[0]), float(pt[1])] for pt in bbox]
            xs = [p[0] for p in poly_points]
            ys = [p[1] for p in poly_points]
            axis_aligned_box = [min(xs), min(ys), max(xs), max(ys)]

            direction = self._compute_text_spatial_info(poly_points, img_w, img_h)

            extracted_items.append({
                "text": cleaned_text,
                "confidence": round(float(conf), 2),
                "polygon": poly_points,
                "box": [round(c, 1) for c in axis_aligned_box],
                "direction": direction,
            })

        # Sort in natural reading order
        ordered_items = self._sort_reading_order(extracted_items)
        return ordered_items

    def annotate(self, image_input: Union[str, bytes, np.ndarray, Image.Image],
                 ocr_results: Optional[List[Dict[str, Any]]] = None) -> Image.Image:
        """
        Draws high-contrast bounding boxes around detected text blocks with green highlighting.
        """
        pil_img = load_image(image_input).copy()
        if ocr_results is None:
            ocr_results = self.read_text(pil_img)

        draw = ImageDraw.Draw(pil_img)

        for item in ocr_results:
            poly = [(p[0], p[1]) for p in item["polygon"]]
            # Draw green polygon border
            draw.polygon(poly, outline=(0, 230, 100), width=3)

            # Draw background tag
            min_x, min_y = item["box"][0], item["box"][1]
            tag_text = item["text"]
            tag_h = 20
            tag_w = len(tag_text) * 8 + 10

            draw.rectangle([min_x, max(0, min_y - tag_h), min_x + tag_w, min_y], fill=(0, 0, 0))
            draw.text((min_x + 4, max(0, min_y - tag_h) + 2), tag_text, fill=(0, 255, 120))

        return pil_img

    def generate_spoken_summary(self, ocr_results: List[Dict[str, Any]]) -> str:
        """
        Converts extracted OCR text into a structured, screen-reader friendly spoken message.
        """
        if not ocr_results:
            return "No readable printed text or signage detected."

        text_lines = [item["text"] for item in ocr_results]
        combined = " ".join(text_lines)

        return f"Detected text reads: \"{combined}\"."
