"""
Standalone audio-to-text transcription using faster-whisper.

Usage:
  python transcribe_combined.py --input audio.m4a  [flags]
  python transcribe_combined.py --input audio.m4a --output transcript.txt --model medium
  python transcribe_combined.py --input audio.wav

Flags:
  --input FILE    Audio file to transcribe (required)
  --output FILE   Output transcript file (default: transcript.txt)
  --model NAME    Whisper model: tiny, base, small, medium, large-v3 (default: large-v3)
  --language CODE Force a language (e.g. "en", "hi") or leave unset for auto-detect
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "meeting-intelligence"))

import config
from faster_whisper import WhisperModel


def main():
    parser = argparse.ArgumentParser(
        description="Standalone audio-to-text transcription",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input", required=True, help="Audio file to transcribe")
    parser.add_argument("--output", default="transcript.txt", help="Output transcript file")
    parser.add_argument("--model", default=config.WHISPER_MODEL, help="Whisper model size")
    parser.add_argument("--language", default=None, help="Force a language code (e.g. en, hi)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Audio file not found: {args.input}")
        sys.exit(1)

    model_name = args.model
    language = args.language or config.WHISPER_LANGUAGE

    print(f"Loading Whisper {model_name} (int8, CPU)...")
    model = WhisperModel(
        model_name,
        device="cpu",
        compute_type=config.WHISPER_COMPUTE_TYPE,
    )

    print(f"Transcribing {args.input} (language={language or 'auto'})...")
    segments_iter, info = model.transcribe(
        str(input_path),
        language=language,
        word_timestamps=True,
        beam_size=5,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )

    detected_lang = getattr(info, "language", "unknown")
    detected_prob = getattr(info, "language_probability", 0.0)
    print(f"Detected language: {detected_lang} (confidence={detected_prob:.2f})")

    output_path = Path(args.output)
    count = 0
    with open(output_path, "w") as f:
        for seg in segments_iter:
            count += 1
            text = seg.text.strip()
            line = f"[{seg.start:.1f}s - {seg.end:.1f}s] {text}"
            f.write(line + "\n")
            f.flush()
            if count % 50 == 0:
                print(f"  ({count} segments)...")

    print(f"\nDone! {count} segments saved to {output_path}")


if __name__ == "__main__":
    main()
