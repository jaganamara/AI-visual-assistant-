"""
Unit tests for TextToSpeech and SpeechRecognizer modules.
"""

import os
import tempfile
import pytest
import numpy as np

from src.speech.text_to_speech import TextToSpeech
from src.speech.speech_recognizer import SpeechRecognizer


def test_tts_audio_bytes_generation():
    """Verify that TextToSpeech generates valid audio bytes for in-browser playback."""
    tts = TextToSpeech()
    test_phrase = "Attention: A chair is directly in front of you."

    audio_bytes = tts.to_audio_bytes(test_phrase)
    assert isinstance(audio_bytes, bytes)
    assert len(audio_bytes) > 500  # Valid MP3 payload


def test_tts_save_audio():
    """Verify that TextToSpeech can save audio files to disk."""
    tts = TextToSpeech()
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tf:
        temp_path = tf.name

    try:
        success = tts.save_audio("Caution, obstacle ahead.", temp_path)
        assert success is True
        assert os.path.exists(temp_path)
        assert os.path.getsize(temp_path) > 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_speech_recognizer_lifecycle_and_transcribe():
    """Verify SpeechRecognizer initializes, transcribes audio, and unloads from memory."""
    recognizer = SpeechRecognizer(model_size="tiny")

    # Generate a 1-second synthetic sine wave audio (silence/tone) to test pipeline
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False, dtype=np.float32)
    audio_data = 0.1 * np.sin(2 * np.pi * 440 * t)  # 440Hz tone

    # Transcribe should handle audio array without throwing error
    transcription = recognizer.transcribe(audio_data)
    assert isinstance(transcription, str)

    # Test unload
    recognizer.unload()
    assert recognizer.model is None
