"""
YOLOv8 Object Detection and Spatial Reasoning Module.
Provides object localization, relative direction estimation (Left, Center, Right),
proximity calculation, high-contrast visualization, and VRAM management.
"""

import os
import logging
from typing import List, Dict, Any, Union, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from ultralytics import YOLO

from src.utils.device_manager import DeviceManager, get_cache_dir
from src.vision.utils import load_image

logger = logging.getLogger(__name__)


class ObjectDetector:
    """
    Object detection engine using pretrained YOLOv8 with spatial localization
    tailored for visually impaired assistance.
    """

    def __init__(self, model_name: str = "yolov8n.pt", confidence_threshold: float = 0.40):
        """
        Initializes the YOLOv8 detector.
        Model weights are stored in the designated models_cache/ directory.
        """
        self.confidence_threshold = confidence_threshold
        self.model_name = model_name
        self.cache_dir = get_cache_dir("yolo")
        self.model_path = os.path.join(self.cache_dir, model_name)
        self.device = DeviceManager.get_preferred_device()
        self.model: Optional[YOLO] = None

    def _ensure_model_loaded(self) -> None:
        """Lazy loader: loads model weights only when first needed."""
        if self.model is None:
            logger.info(f"Loading YOLO model '{self.model_name}' on device '{self.device}'...")
            try:
                # ultralytics automatically downloads if model_path is not found
                self.model = YOLO(self.model_name)
                # Ensure local checkpoint copy is maintained in cache directory
                if hasattr(self.model, 'ckpt_path') and os.path.exists(self.model.ckpt_path):
                    import shutil
                    if not os.path.exists(self.model_path) and os.path.abspath(self.model.ckpt_path) != os.path.abspath(self.model_path):
                        shutil.copy(self.model.ckpt_path, self.model_path)
            except Exception as e:
                logger.error(f"Failed to load YOLO on {self.device}: {e}. Retrying on CPU.")
                self.device = torch.device("cpu")
                self.model = YOLO(self.model_name)

    def unload(self) -> None:
        """Explicitly unloads model from VRAM to preserve memory for other modules."""
        if self.model is not None:
            logger.info("Unloading YOLO model from memory...")
            del self.model
            self.model = None
            DeviceManager.clear_vram_cache()

    @staticmethod
    def _compute_spatial_position(box: List[float], img_w: int, img_h: int) -> Dict[str, str]:
        """
        Derives directional position and proximity alert based on normalized bounding box geometry.
        """
        x1, y1, x2, y2 = box
        cx = ((x1 + x2) / 2.0) / img_w
        cy = ((y1 + y2) / 2.0) / img_h
        area_ratio = ((x2 - x1) * (y2 - y1)) / (img_w * img_h)

        # Horizontal classification
        if cx < 0.35:
            horizontal = "on your left"
        elif cx > 0.65:
            horizontal = "on your right"
        else:
            horizontal = "directly in front"

        # Proximity estimation based on area percentage
        if area_ratio >= 0.20:
            proximity = "immediate obstacle (very close)"
            urgency = "HIGH"
        elif area_ratio >= 0.06:
            proximity = "nearby"
            urgency = "MEDIUM"
        else:
            proximity = "in the distance"
            urgency = "LOW"

        return {
            "direction": horizontal,
            "proximity": proximity,
            "urgency": urgency,
            "area_percentage": round(area_ratio * 100, 1),
        }

    def detect(self, image_input: Union[str, bytes, np.ndarray, Image.Image]) -> List[Dict[str, Any]]:
        """
        Runs object detection on an image and returns structured detections with spatial data.
        """
        pil_img = load_image(image_input)
        img_w, img_h = pil_img.size

        self._ensure_model_loaded()

        def _run_inference(dev_str: str):
            # Ultralytics accepts device parameter as 'cuda:0' or 'cpu'
            device_target = 0 if "cuda" in dev_str else "cpu"
            results = self.model.predict(
                source=pil_img,
                conf=self.confidence_threshold,
                device=device_target,
                verbose=False,
            )
            return results

        try:
            results = _run_inference(str(self.device))
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            logger.warning(f"CUDA execution issue ({e}). Clearing VRAM and falling back to CPU...")
            DeviceManager.clear_vram_cache()
            self.device = torch.device("cpu")
            results = _run_inference("cpu")

        detections = []
        if not results:
            return detections

        first_res = results[0]
        boxes = first_res.boxes

        if boxes is None or len(boxes) == 0:
            return detections

        for box in boxes:
            xyxy = box.xyxy[0].tolist()
            conf = float(box.conf[0].item())
            cls_id = int(box.cls[0].item())
            class_name = self.model.names.get(cls_id, f"object_{cls_id}")

            spatial_info = self._compute_spatial_position(xyxy, img_w, img_h)

            detections.append({
                "label": class_name,
                "confidence": round(conf, 2),
                "box": [round(coord, 1) for coord in xyxy],
                "direction": spatial_info["direction"],
                "proximity": spatial_info["proximity"],
                "urgency": spatial_info["urgency"],
                "area_percentage": spatial_info["area_percentage"],
            })

        # Sort detections: highest urgency and highest area first (closest hazards prioritized)
        detections.sort(key=lambda d: (d["urgency"] == "HIGH", d["area_percentage"]), reverse=True)
        return detections

    def annotate(self, image_input: Union[str, bytes, np.ndarray, Image.Image],
                 detections: Optional[List[Dict[str, Any]]] = None) -> Image.Image:
        """
        Draws high-contrast accessible bounding boxes and directional text badges on the image.
        """
        pil_img = load_image(image_input).copy()
        if detections is None:
            detections = self.detect(pil_img)

        draw = ImageDraw.Draw(pil_img)

        # Palette for high-contrast accessibility: Amber, Cyan, Lime, Magenta
        colors = {
            "HIGH": (255, 60, 60),      # Bright Red for high urgency
            "MEDIUM": (255, 180, 0),    # Amber for medium
            "LOW": (50, 200, 255),      # Bright Cyan for low
        }

        for det in detections:
            x1, y1, x2, y2 = det["box"]
            urgency = det.get("urgency", "LOW")
            color = colors.get(urgency, (255, 255, 0))

            # Draw thick bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            # Label badge
            label_text = f"{det['label'].upper()} ({det['direction']})"
            badge_h = 24
            badge_w = len(label_text) * 8 + 12

            # Background pill for readability
            draw.rectangle([x1, max(0, y1 - badge_h), x1 + badge_w, y1], fill=(0, 0, 0))
            draw.text((x1 + 6, max(0, y1 - badge_h) + 4), label_text, fill=color)

        return pil_img

    def generate_spoken_summary(self, detections: List[Dict[str, Any]]) -> str:
        """
        Converts detection results into concise, screen-reader friendly spoken text.
        """
        if not detections:
            return "No prominent objects or obstacles detected in view."

        lines = []
        # Group by direction
        front_objs = [f"{d['label']} ({d['proximity']})" for d in detections if "in front" in d["direction"]]
        left_objs = [d['label'] for d in detections if "left" in d["direction"]]
        right_objs = [d['label'] for d in detections if "right" in d["direction"]]

        if front_objs:
            lines.append(f"Directly in front: {', '.join(front_objs)}.")
        if left_objs:
            lines.append(f"On your left: {', '.join(left_objs)}.")
        if right_objs:
            lines.append(f"On your right: {', '.join(right_objs)}.")

        return " ".join(lines)
