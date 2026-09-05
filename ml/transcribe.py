# ml/transcribe.py
# Standalone test script: transcribes a single audio file using faster-whisper.
# Run this directly to confirm speech-to-text works before we wire it into Flask.

import os
import sys
from faster_whisper import WhisperModel

def transcribe_audio(audio_path, model_size="base"):
    """
    Transcribes the given audio file and returns a list of segments.
    Each segment has: start time, end time, and text.

    model_size options (bigger = more accurate but slower):
      "tiny"   - fastest, least accurate
      "base"   - good balance for a prototype (recommended to start)
      "small"  - more accurate, slower
      "medium" - much slower on CPU, only if you have a good machine
    """

    print(f"[INFO] Loading Whisper model: {model_size} (first run downloads it, please wait)...")

    # device="cpu" and compute_type="int8" keep this fast and free (no GPU needed)
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    print(f"[INFO] Transcribing: {audio_path}")
    segments, info = model.transcribe(audio_path, beam_size=5)

    print(f"[INFO] Detected language: {info.language} (confidence: {info.language_probability:.2f})")
    print("-" * 50)

    results = []
    for segment in segments:
        text = segment.text.strip()
        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s]  {text}")
        results.append({
            "start": segment.start,
            "end": segment.end,
            "text": text
        })

    return results


if __name__ == "__main__":
    # Usage: python ml/transcribe.py path/to/audio.webm
    if len(sys.argv) < 2:
        print("Usage: python ml/transcribe.py <path_to_audio_file>")
        sys.exit(1)

    audio_file = sys.argv[1]

    if not os.path.exists(audio_file):
        print(f"[ERROR] File not found: {audio_file}")
        sys.exit(1)

    results = transcribe_audio(audio_file)

    print("-" * 50)
    print(f"[INFO] Done. {len(results)} segments transcribed.")