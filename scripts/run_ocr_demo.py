"""
Demonstration script for OCR Text Extraction.
Extracts printed text, signage, and labels, sorts them in natural reading order,
and generates spoken accessibility audio descriptions.
"""

import sys
import os
from PIL import Image, ImageDraw

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.ocr.ocr_reader import OCRReader
from src.vision.utils import load_image


def create_sample_signage_image() -> Image.Image:
    """Generates a sample store/office sign image for demonstration."""
    img = Image.new("RGB", (700, 350), color=(250, 250, 250))
    draw = ImageDraw.Draw(img)

    # Header / Notice Banner
    draw.rectangle([40, 30, 660, 90], fill=(20, 50, 150))
    draw.text((60, 48), "MAIN ENTRANCE - BUILDING B", fill=(255, 255, 255))

    # Room number
    draw.rectangle([40, 120, 300, 200], fill=(240, 240, 240), outline=(0, 0, 0), width=2)
    draw.text((60, 145), "ROOM 204", fill=(0, 0, 0))

    # Caution sign
    draw.rectangle([340, 120, 660, 200], fill=(255, 230, 0), outline=(0, 0, 0), width=2)
    draw.text((360, 145), "CAUTION: WET FLOOR", fill=(0, 0, 0))

    # Hours
    draw.text((50, 260), "Visiting Hours: 9:00 AM - 5:00 PM", fill=(40, 40, 40))

    return img


def main():
    print("=" * 65)
    print("AI-POWERED VISUAL ASSISTANT: OCR TEXT EXTRACTION")
    print("=" * 65)

    reader = OCRReader(confidence_threshold=0.25)

    # Determine image source
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        image_path = sys.argv[1]
        print(f"Loading input image: {image_path}")
        img = load_image(image_path)
    else:
        print("No image provided as argument. Generating sample signage image...")
        img = create_sample_signage_image()

    print("\nRunning OCR text detection and recognition...")
    results = reader.read_text(img)

    print(f"\nExtracted {len(results)} text block(s) in reading order:")
    for idx, item in enumerate(results, 1):
        print(f"  {idx}. \"{item['text']}\" (conf: {item['confidence']*100:.0f}%, {item['direction']})")
        print(f"     Box: {item['box']}")

    spoken_summary = reader.generate_spoken_summary(results)
    print("\nSpoken Accessibility Output:")
    print(f"  {spoken_summary}")

    # Save visual annotation
    output_path = os.path.join(PROJECT_ROOT, "ocr_result.jpg")
    annotated_img = reader.annotate(img, results)
    annotated_img.save(output_path)
    print(f"\nAnnotated image saved to: {output_path}")

    # Explicitly unload to preserve VRAM
    reader.unload()
    print("OCR model cleanly unloaded from memory.")
    print("=" * 65)


if __name__ == "__main__":
    main()
