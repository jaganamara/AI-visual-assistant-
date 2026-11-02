"""
Vision Utilities.
Image validation, color format conversion, and synthetic test image generators.
"""

import os
from typing import Union, Tuple
import numpy as np
from PIL import Image, ImageDraw


def load_image(image_input: Union[str, bytes, np.ndarray, Image.Image]) -> Image.Image:
    """
    Safely converts diverse image input types into a standardized RGB PIL Image.
    """
    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")

    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Image path does not exist: {image_input}")
        return Image.open(image_input).convert("RGB")

    if isinstance(image_input, bytes):
        import io
        return Image.open(io.BytesIO(image_input)).convert("RGB")

    if isinstance(image_input, np.ndarray):
        # Handle OpenCV BGR vs RGB
        if len(image_input.shape) == 2:
            return Image.fromarray(image_input).convert("RGB")
        elif image_input.shape[2] == 3:
            return Image.fromarray(image_input).convert("RGB")
        elif image_input.shape[2] == 4:
            return Image.fromarray(image_input).convert("RGB")
        else:
            raise ValueError(f"Unsupported numpy image shape: {image_input.shape}")

    raise TypeError(f"Unsupported image input type: {type(image_input)}")


def create_synthetic_test_image(width: int = 640, height: int = 480) -> Image.Image:
    """
    Generates a high-contrast synthetic test image containing distinct colored shapes
    for testing without needing real camera feeds.
    """
    img = Image.new("RGB", (width, height), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)

    # Left object (blue rectangle)
    draw.rectangle([50, 150, 200, 350], fill=(30, 80, 200), outline=(0, 0, 0), width=3)
    draw.text((60, 160), "TEST OBJECT A", fill=(255, 255, 255))

    # Center object (red circle/ellipse)
    draw.ellipse([250, 120, 390, 380], fill=(220, 50, 50), outline=(0, 0, 0), width=3)
    draw.text((270, 200), "TEST OBJECT B", fill=(255, 255, 255))

    # Right object (green rectangle)
    draw.rectangle([440, 180, 590, 360], fill=(40, 160, 60), outline=(0, 0, 0), width=3)
    draw.text((450, 190), "TEST OBJECT C", fill=(255, 255, 255))

    return img
