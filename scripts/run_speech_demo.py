"""
Demonstration script for Speech Interaction (STT and TTS).
Demonstrates text-to-speech audio synthesis and speech recognition.
"""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.speech.text_to_speech import TextToSpeech
from src.speech.speech_recognizer import SpeechRecognizer


def main():
    print("=" * 65)
    print("AI-POWERED VISUAL ASSISTANT: SPEECH INTERACTION (STT & TTS)")
    print("=" * 65)

    tts = TextToSpeech()
    recognizer = SpeechRecognizer(model_size="tiny")

    # 1. Text-to-Speech Demonstration
    sample_warning = (
        "Caution: Immediate obstacle directly in front. "
        "A bus is approaching. Obstacle area is 47 percent."
    )
    print("\n[1/3] Synthesizing speech...")
    print(f"  Text: \"{sample_warning}\"")

    audio_file = os.path.join(PROJECT_ROOT, "speech_sample.mp3")
    tts.save_audio(sample_warning, audio_file)
    print(f"  -> Generated MP3 audio saved to: {audio_file}")

    print("  -> Playing audio through laptop speakers (pyttsx3 / SAPI5)...")
    tts.speak(sample_warning)

    # 2. Speech-to-Text Demonstration
    print("\n[2/3] Transcribing generated audio via Whisper STT...")
    transcription = recognizer.transcribe(audio_file)
    print(f"  -> Whisper Transcription: \"{transcription}\"")

    # 3. Memory Cleanup
    print("\n[3/3] Cleaning up...")
    recognizer.unload()
    print("Whisper model cleanly unloaded from memory.")
    print("=" * 65)


if __name__ == "__main__":
    main()
