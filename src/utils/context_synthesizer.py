"""
Context Synthesizer Module.
Aggregates multimodal signals (YOLO detections, OCR text, and BLIP caption)
into an accessible, safety-prioritized spoken narrative for visually impaired users.
"""

from typing import List, Dict, Any, Optional


class ContextSynthesizer:
    """
    Synthesizes vision, OCR, and vision-language outputs into a prioritized narrative.
    Safety-critical alerts (immediate obstacles) are prioritized first.
    """

    @staticmethod
    def synthesize(caption: Optional[str] = None,
                   detections: Optional[List[Dict[str, Any]]] = None,
                   ocr_results: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Synthesizes multimodal analysis into structured sections and a unified spoken string.
        """
        sections = []
        immediate_hazards = []
        nearby_objects = []

        # 1. Process YOLO Detections & Safety Proximity
        if detections:
            for d in detections:
                label = d["label"]
                direction = d["direction"]
                urgency = d.get("urgency", "LOW")
                proximity = d.get("proximity", "in the distance")

                if urgency == "HIGH":
                    immediate_hazards.append(f"{label} ({direction})")
                else:
                    nearby_objects.append(f"{label} {direction}")

        # 2. Priority 1: Hazard / Obstacle Alert
        if immediate_hazards:
            hazard_str = f"Caution: Immediate obstacle close ahead: {', '.join(immediate_hazards)}."
            sections.append(hazard_str)

        # 3. Priority 2: High-Level Scene Overview (BLIP Caption)
        if caption and caption.strip():
            clean_caption = caption.strip()
            # Capitalize first letter
            clean_caption = clean_caption[0].upper() + clean_caption[1:]
            if not clean_caption.endswith("."):
                clean_caption += "."
            sections.append(f"Scene view: {clean_caption}")

        # 4. Priority 3: Spatial Objects of Interest (Non-hazards)
        if nearby_objects:
            # Dedup while preserving order
            unique_objects = list(dict.fromkeys(nearby_objects))
            if len(unique_objects) <= 5:
                objs_str = f"Objects present: {', '.join(unique_objects)}."
            else:
                objs_str = f"Objects present: {', '.join(unique_objects[:5])}, and others."
            sections.append(objs_str)

        # 5. Priority 4: Printed Signs & Text (EasyOCR)
        if ocr_results:
            text_blocks = [item["text"] for item in ocr_results if len(item["text"].strip()) > 1]
            if text_blocks:
                joined_text = " ".join(text_blocks)
                sections.append(f"Detected text reads: \"{joined_text}\".")

        if not sections:
            unified_narrative = "The scene appears clear. No prominent objects or text were identified."
        else:
            unified_narrative = " ".join(sections)

        return {
            "immediate_hazards": immediate_hazards,
            "scene_caption": caption or "",
            "nearby_objects": nearby_objects,
            "ocr_text": " ".join([i["text"] for i in (ocr_results or [])]),
            "spoken_narrative": unified_narrative,
        }
