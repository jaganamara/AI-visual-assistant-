"""
Speech Recognition (STT) Module using OpenAI Whisper.
Transcribes spoken user queries from audio recordings into text.
Uses soundfile and pure Python audio resampling to eliminate the need for system-level ffmpeg.
"""

import os
import io
import logging
from typing import Union, Optional
import numpy as np
import torch
import soundfile as sf
from scipy import signal

from src.utils.device_manager import DeviceManager, get_cache_dir

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16000


def _load_audio_to_16k_mono(audio_input: Union[str, bytes, np.ndarray]) -> np.ndarray:
    """
    Decodes audio from path, bytes, or numpy array to a 1D float32 array at 16000 Hz,
    bypassing the need for external ffmpeg binary installation.
    """
    if isinstance(audio_input, np.ndarray):
        data = audio_input.astype(np.float32)
        sr = TARGET_SAMPLE_RATE
    elif isinstance(audio_input, bytes):
        data, sr = sf.read(io.BytesIO(audio_input), dtype="float32")
    elif isinstance(audio_input, str):
        if not os.path.exists(audio_input):
            raise FileNotFoundError(f"Audio file not found: {audio_input}")
        data, sr = sf.read(audio_input, dtype="float32")
    else:
        raise TypeError(f"Unsupported audio input type: {type(audio_input)}")

    # Convert multi-channel / stereo to mono
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    # Resample to 16000 Hz if needed
    if sr != TARGET_SAMPLE_RATE:
        num_target_samples = int(len(data) * TARGET_SAMPLE_RATE / sr)
        data = signal.resample(data, num_target_samples).astype(np.float32)

    return data


class SpeechRecognizer:
    """
    Speech-to-Text engine using OpenAI Whisper (tiny/base).
    Features local model caching, lazy loading, soundfile ingestion (no ffmpeg required),
    and automatic CPU fallback.
    """

    def __init__(self, model_size: str = "tiny"):
        self.model_size = model_size
        self.cache_dir = get_cache_dir("whisper")
        self.device = DeviceManager.get_preferred_device()
        self.model = None

    def _ensure_model_loaded(self) -> None:
        """Lazy loader: loads Whisper model weights only when needed."""
        if self.model is None:
            logger.info(f"Loading Whisper model '{self.model_size}' on {self.device}...")
            try:
                import whisper
                self.model = whisper.load_model(
                    self.model_size,
                    device=self.device,
                    download_root=self.cache_dir,
                )
            except (torch.cuda.OutOfMemoryError, RuntimeError, Exception) as e:
                logger.warning(f"Failed to load Whisper on {self.device} ({e}). Falling back to CPU...")
                DeviceManager.clear_vram_cache()
                self.device = torch.device("cpu")
                import whisper
                self.model = whisper.load_model(
                    self.model_size,
                    device="cpu",
                    download_root=self.cache_dir,
                )

    def unload(self) -> None:
        """Unloads model weights from memory and clears VRAM."""
        if self.model is not None:
            logger.info("Unloading Whisper model from memory...")
            del self.model
            self.model = None
            DeviceManager.clear_vram_cache()

    def transcribe(self, audio_input: Union[str, bytes, np.ndarray],
                   language: Optional[str] = "en") -> str:
        """
        Transcribes speech from an audio file path, raw audio bytes, or numpy audio array.
        """
        self._ensure_model_loaded()

        # Preprocess directly to 16kHz float32 numpy array (no ffmpeg needed!)
        audio_array = _load_audio_to_16k_mono(audio_input)

        use_fp16 = (self.device.type == "cuda")

        try:
            result = self.model.transcribe(
                audio_array,
                language=language,
                fp16=use_fp16,
            )
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            logger.warning(f"CUDA OOM during Whisper transcription ({e}). Retrying on CPU...")
            self.unload()
            self.device = torch.device("cpu")
            self._ensure_model_loaded()
            result = self.model.transcribe(
                audio_array,
                language=language,
                fp16=False,
            )

        transcribed_text = result.get("text", "").strip()
        return transcribed_text
