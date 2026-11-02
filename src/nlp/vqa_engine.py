"""
Visual Question Answering (VQA) Module using Salesforce BLIP VQA Base.
Answers open-ended natural language questions conditioned on visual scene features.
"""

import os
import logging
from typing import Union
from PIL import Image
import torch

from src.utils.device_manager import DeviceManager, get_cache_dir
from src.vision.utils import load_image

logger = logging.getLogger(__name__)


class VQAEngine:
    """
    Visual Question Answering engine conditioned on image features and text queries.
    Enables visually impaired users to query specific visual details (e.g. 'What color is the bag?').
    """

    MODEL_ID = "Salesforce/blip-vqa-base"

    def __init__(self):
        self.cache_dir = get_cache_dir("huggingface")
        os.environ["HF_HOME"] = self.cache_dir
        self.device = DeviceManager.get_preferred_device()
        self.dtype = DeviceManager.get_optimal_dtype(self.device)
        self.processor = None
        self.model = None

    def _ensure_model_loaded(self) -> None:
        """Lazy loader: loads BLIP VQA model only when invoked."""
        if self.model is None or self.processor is None:
            logger.info(f"Loading BLIP VQA Engine on {self.device} with {self.dtype}...")
            try:
                from transformers import BlipProcessor, BlipForQuestionAnswering

                self.processor = BlipProcessor.from_pretrained(
                    self.MODEL_ID,
                    cache_dir=self.cache_dir,
                )
                self.model = BlipForQuestionAnswering.from_pretrained(
                    self.MODEL_ID,
                    torch_dtype=self.dtype,
                    cache_dir=self.cache_dir,
                ).to(self.device)
                self.model.eval()
            except (torch.cuda.OutOfMemoryError, RuntimeError, Exception) as e:
                logger.warning(f"Failed to load BLIP VQA on {self.device} ({e}). Falling back to CPU...")
                DeviceManager.clear_vram_cache()
                self.device = torch.device("cpu")
                self.dtype = torch.float32

                from transformers import BlipProcessor, BlipForQuestionAnswering
                self.processor = BlipProcessor.from_pretrained(self.MODEL_ID, cache_dir=self.cache_dir)
                self.model = BlipForQuestionAnswering.from_pretrained(
                    self.MODEL_ID,
                    torch_dtype=torch.float32,
                    cache_dir=self.cache_dir,
                ).to("cpu")
                self.model.eval()

    def unload(self) -> None:
        """Unloads model weights and evicts VRAM allocation."""
        if self.model is not None:
            logger.info("Unloading BLIP VQA Engine from memory...")
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            DeviceManager.clear_vram_cache()

    def answer_question(self, image_input: Union[str, bytes, Image.Image], question: str) -> str:
        """
        Answers a natural language question about an image.
        """
        if not question or not question.strip():
            return "Please provide a valid question."

        pil_img = load_image(image_input)
        cleaned_question = question.strip()
        self._ensure_model_loaded()

        def _infer(dev: torch.device, dt: torch.dtype):
            inputs = self.processor(pil_img, cleaned_question, return_tensors="pt").to(dev, dt)
            with torch.inference_mode():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=30,
                    num_beams=3,
                )
            answer = self.processor.decode(out[0], skip_special_tokens=True).strip()
            return answer

        try:
            return _infer(self.device, self.dtype)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            logger.warning(f"CUDA OOM during VQA inference ({e}). Falling back to CPU...")
            self.unload()
            self.device = torch.device("cpu")
            self.dtype = torch.float32
            self._ensure_model_loaded()
            return _infer(self.device, self.dtype)
