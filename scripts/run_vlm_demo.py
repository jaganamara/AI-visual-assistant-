"""
Demonstration script for BLIP Scene Captioning and Visual Question Answering (VQA).
Generates contextual scene descriptions, answers visual questions, and synthesizes
multimodal information.
"""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.nlp.scene_captioner import SceneCaptioner
from src.nlp.vqa_engine import VQAEngine
from src.vision.object_detector import ObjectDetector
from src.ocr.ocr_reader import OCRReader
from src.utils.context_synthesizer import ContextSynthesizer
from src.vision.utils import load_image


def main():
    print("=" * 65)
    print("AI-POWERED VISUAL ASSISTANT: SCENE UNDERSTANDING & VQA DEMO")
    print("=" * 65)

    # Determine image source
    default_img = os.path.join(PROJECT_ROOT, "sample_images", "bus.jpg")
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        image_path = sys.argv[1]
    else:
        image_path = default_img

    print(f"Loading input scene: {image_path}")
    img = load_image(image_path)

    # 1. Run YOLO Object Detection
    print("\n[1/4] Running YOLOv8 Object Detection...")
    detector = ObjectDetector(model_name="yolov8n.pt", confidence_threshold=0.35)
    detections = detector.detect(img)
    detector.unload()  # Unload immediately to preserve VRAM for BLIP

    # 2. Run EasyOCR
    print("[2/4] Running EasyOCR Text Recognition...")
    ocr = OCRReader(confidence_threshold=0.25)
    ocr_results = ocr.read_text(img)
    ocr.unload()  # Unload immediately to preserve VRAM for BLIP

    # 3. Run BLIP Scene Captioning
    print("[3/4] Running BLIP Scene Captioning...")
    captioner = SceneCaptioner()
    caption = captioner.generate_caption(img)
    print(f"  -> Generated Caption: \"{caption}\"")
    captioner.unload()

    # 4. Synthesize Unified Spoken Output
    print("\n[4/4] Synthesizing Holistic Spoken Narrative...")
    synthesis = ContextSynthesizer.synthesize(
        caption=caption,
        detections=detections,
        ocr_results=ocr_results
    )
    print("\n" + "-" * 65)
    print("UNIFIED ASSISTIVE SPOKEN SUMMARY:")
    print(f"  \"{synthesis['spoken_narrative']}\"")
    print("-" * 65)

    # 5. Interactive Visual Question Answering
    print("\n[VQA Engine] Testing Visual Question Answering...")
    vqa = VQAEngine()

    questions = [
        "What vehicle is in the scene?",
        "Are there people present?",
        "What is the color of the bus?",
    ]

    for q in questions:
        ans = vqa.answer_question(img, q)
        print(f"  Q: {q}")
        print(f"  A: {ans}\n")

    vqa.unload()
    print("All models cleanly unloaded from memory.")
    print("=" * 65)


if __name__ == "__main__":
    main()
