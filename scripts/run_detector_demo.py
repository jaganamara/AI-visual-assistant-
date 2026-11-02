"""
Demonstration script for YOLOv8 Object Detection and Spatial Reasoning.
Can test with either an input image file or a generated synthetic test scene.
"""

import sys
import os
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.vision.object_detector import ObjectDetector
from src.vision.utils import create_synthetic_test_image, load_image


def main():
    print("=" * 65)
    print("AI-POWERED VISUAL ASSISTANT: OBJECT DETECTION & SPATIAL REASONING")
    print("=" * 65)

    detector = ObjectDetector(model_name="yolov8n.pt", confidence_threshold=0.30)

    # Determine image source
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        image_path = sys.argv[1]
        print(f"Loading input image: {image_path}")
        img = load_image(image_path)
    else:
        print("No image provided as argument. Generating synthetic test scene...")
        img = create_synthetic_test_image(640, 480)

    print("\nRunning inference...")
    detections = detector.detect(img)

    print(f"\nDetected {len(detections)} object(s):")
    for idx, d in enumerate(detections, 1):
        print(f"  {idx}. [{d['urgency']}] {d['label'].upper()} (conf: {d['confidence']*100:.0f}%)")
        print(f"     Direction: {d['direction']} | Proximity: {d['proximity']} | Area: {d['area_percentage']}%")
        print(f"     Bounding Box: {d['box']}")

    spoken_summary = detector.generate_spoken_summary(detections)
    print("\nSpoken Accessibility Output:")
    print(f"  \"{spoken_summary}\"")

    # Save annotated visual output
    output_path = os.path.join(PROJECT_ROOT, "detection_result.jpg")
    annotated_img = detector.annotate(img, detections)
    annotated_img.save(output_path)
    print(f"\nAnnotated image saved to: {output_path}")

    # Explicitly unload to demonstrate VRAM safety
    detector.unload()
    print("Model cleanly unloaded from memory.")
    print("=" * 65)


if __name__ == "__main__":
    main()
