"""
Text-to-Speech (TTS) Module.
Provides dual-mode audio generation:
1. Local offline playback via Windows SAPI5 (pyttsx3)
2. In-memory / on-disk MP3 audio generation via gTTS (for Streamlit in-browser playback)
"""

import io
import os
import logging
from typing import Optional, Union

logger = logging.getLogger(__name__)


class TextToSpeech:
    """
    Dual-engine Text-to-Speech synthesizer.
    """

    def __init__(self, rate: int = 175, volume: float = 1.0):
        self.rate = rate
        self.volume = volume
        self._sapi_engine = None

    def _get_sapi_engine(self):
        """Initializes pyttsx3 engine safely."""
        if self._sapi_engine is None:
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.setProperty("rate", self.rate)
                engine.setProperty("volume", self.volume)
                self._sapi_engine = engine
            except Exception as e:
                logger.warning(f"Could not initialize offline pyttsx3 engine ({e}).")
                self._sapi_engine = None
        return self._sapi_engine

    def speak(self, text: str) -> bool:
        """
        Speaks text aloud immediately through laptop speakers using Windows SAPI5.
        Returns True on success, False if offline speech engine is unavailable.
        """
        if not text or not text.strip():
            return False

        engine = self._get_sapi_engine()
        if engine is not None:
            try:
                engine.say(text)
                engine.runAndWait()
                return True
            except Exception as e:
                logger.warning(f"pyttsx3 playback encountered error: {e}")
                return False
        return False

    def to_audio_bytes(self, text: str, lang: str = "en") -> bytes:
        """
        Synthesizes text into MP3 audio bytes using gTTS.
        Perfect for rendering inside Streamlit's st.audio() widget.
        """
        if not text or not text.strip():
            return b""

        try:
            from gtts import gTTS
            tts = gTTS(text=text.strip(), lang=lang, slow=False)
            fp = io.BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            return fp.read()
        except Exception as e:
            logger.warning(f"gTTS audio synthesis failed ({e}).")
            return b""

    def save_audio(self, text: str, file_path: str, lang: str = "en") -> bool:
        """
        Saves synthesized speech to an audio file on disk (.mp3 or .wav).
        """
        audio_bytes = self.to_audio_bytes(text, lang=lang)
        if audio_bytes:
            with open(file_path, "wb") as f:
                f.write(audio_bytes)
            return True

        # Fallback to pyttsx3 save
        engine = self._get_sapi_engine()
        if engine is not None:
            try:
                engine.save_to_file(text, file_path)
                engine.runAndWait()
                return True
            except Exception as e:
                logger.error(f"Failed to save audio file with pyttsx3 ({e})")
                return False

        return False
