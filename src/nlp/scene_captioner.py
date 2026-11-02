"""
Scene Captioning Module using Salesforce BLIP (Bootstrapping Language-Image Pre-training).
Generates contextual natural language descriptions of the camera view or uploaded images.
"""

import os
import logging
from typing import Union, Optional
from PIL import Image
import torch

from src.utils.device_manager import DeviceManager, get_cache_dir
from src.vision.utils import load_image

logger = logging.getLogger(__name__)


class SceneCaptioner:
    """
    Multimodal scene understanding engine using Salesforce BLIP Image Captioning Base.
    Features FP16 acceleration, memory-safe lazy loading, and automatic CPU fallback.
    """

    MODEL_ID = "Salesforce/blip-image-captioning-base"

    def __init__(self):
        self.cache_dir = get_cache_dir("huggingface")
        os.environ["HF_HOME"] = self.cache_dir
        self.device = DeviceManager.get_preferred_device()
        self.dtype = DeviceManager.get_optimal_dtype(self.device)
        self.processor = None
        self.model = None

    def _ensure_model_loaded(self) -> None:
        """Lazy loader: loads BLIP captioner only when needed."""
        if self.model is None or self.processor is None:
            logger.info(f"Loading BLIP Captioner on {self.device} with {self.dtype}...")
            try:
                from transformers import BlipProcessor, BlipForConditionalGeneration

                self.processor = BlipProcessor.from_pretrained(
                    self.MODEL_ID,
                    cache_dir=self.cache_dir,
                )
                self.model = BlipForConditionalGeneration.from_pretrained(
                    self.MODEL_ID,
                    torch_dtype=self.dtype,
                    cache_dir=self.cache_dir,
                ).to(self.device)
                self.model.eval()
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                logger.warning(f"Failed to load BLIP on {self.device} ({e}). Falling back to CPU...")
                DeviceManager.clear_vram_cache()
                self.device = torch.device("cpu")
                self.dtype = torch.float32

                from transformers import BlipProcessor, BlipForConditionalGeneration
                self.processor = BlipProcessor.from_pretrained(self.MODEL_ID, cache_dir=self.cache_dir)
                self.model = BlipForConditionalGeneration.from_pretrained(
                    self.MODEL_ID,
                    torch_dtype=torch.float32,
                    cache_dir=self.cache_dir,
                ).to("cpu")
                self.model.eval()

    def unload(self) -> None:
        """Unloads model weights and evicts VRAM allocation."""
        if self.model is not None:
            logger.info("Unloading BLIP Captioner from memory...")
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            DeviceManager.clear_vram_cache()

    def generate_caption(self, image_input: Union[str, bytes, Image.Image],
                         prompt: Optional[str] = None,
                         max_new_tokens: int = 50) -> str:
        """
        Generates a natural-language description of an image.
        If prompt is given (e.g. 'a photo of'), generates conditioned captioning.
        """
        pil_img = load_image(image_input)
        self._ensure_model_loaded()

        def _infer(dev: torch.device, dt: torch.dtype):
            if prompt:
                inputs = self.processor(pil_img, prompt, return_tensors="pt").to(dev, dt)
            else:
                inputs = self.processor(pil_img, return_tensors="pt").to(dev, dt)

            with torch.inference_mode():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    num_beams=3,
                    min_length=5,
                    repetition_penalty=1.2,
                )
            caption = self.processor.decode(out[0], skip_special_tokens=True).strip()
            return caption

        try:
            return _infer(self.device, self.dtype)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            logger.warning(f"CUDA OOM during caption generation ({e}). Falling back to CPU...")
            self.unload()
            self.device = torch.device("cpu")
            self.dtype = torch.float32
            self._ensure_model_loaded()
            return _infer(self.device, self.dtype)
