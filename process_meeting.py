"""
Full meeting pipeline CLI: transcribe + diarize + align + analyze.

Usage:
  python process_meeting.py audio1.m4a [audio2.m4a ...]  [flags]
  python process_meeting.py recording.mp3 --speakers 3 --formats md json pdf
  python process_meeting.py *.m4a --output my_outputs

Flags:
  --output DIR    Output directory (default: "outputs")
  --speakers N    Hint for number of speakers (default: 2, 0 = auto-detect)
  --formats F...  Output formats: md, json, pdf, txt (default: md txt)
  --hf-token KEY  HuggingFace token (default: $HF_TOKEN env var)
"""

import argparse
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "meeting-intelligence"))

import config


def concat_to_wav(files: list[str]) -> str:
    if len(files) == 1:
        wav_path = Path(files[0])
        if wav_path.suffix.lower() == ".wav":
            return str(wav_path)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        subprocess.run(
            ["ffmpeg", "-y", "-i", files[0], "-ar", "16000", "-ac", "1", "-f", "wav", tmp.name],
            capture_output=True, text=True, check=True,
        )
        return tmp.name

    concat_list = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
    for f in files:
        concat_list.write(f"file '{Path(f).resolve()}'\n")
    concat_list.close()

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()

    print(f"Concatenating {len(files)} files...")
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list.name,
         "-ar", "16000", "-ac", "1", "-f", "wav", tmp.name],
        capture_output=True, text=True, check=True,
    )
    Path(concat_list.name).unlink()
    return tmp.name


def parse_formats(args_formats: list[str]) -> list[str]:
    mapping = {
        "md": "Markdown", "markdown": "Markdown",
        "json": "JSON",
        "pdf": "PDF",
        "txt": "Plain Text", "text": "Plain Text", "plain": "Plain Text",
    }
    return [mapping[f.lower()] for f in args_formats if f.lower() in mapping]


def main():
    parser = argparse.ArgumentParser(
        description="Full meeting pipeline: transcribe + diarize + align + analyze",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("files", nargs="+", help="Audio file(s) to process")
    parser.add_argument("--output", default="outputs", help="Output directory")
    parser.add_argument("--speakers", type=int, default=2, help="Number of speakers (0 = auto)")
    parser.add_argument("--formats", nargs="+", default=["md", "txt"], help="Output formats: md, json, pdf, txt")
    parser.add_argument("--hf-token", default="", help="HuggingFace token")
    args = parser.parse_args()

    hf_token = args.hf_token or config.HF_TOKEN
    if not hf_token:
        print("ERROR: HuggingFace token required. Set HF_TOKEN env var or use --hf-token.")
        print("Get a token at: https://huggingface.co/settings/tokens")
        sys.exit(1)

    config.OUTPUT_DIR = args.output
    formats = parse_formats(args.formats)
    num_speakers = args.speakers if args.speakers > 0 else None

    wav_path = None
    try:
        print("=" * 60)
        print("MEETING INTELLIGENCE — Full Pipeline")
        print("=" * 60)

        wav_path = concat_to_wav(args.files)
        print(f"Audio prepared: {wav_path}")

        print("\n[1/5] Transcribing...")
        import transcriber
        segments = transcriber.transcribe(wav_path)
        if not segments:
            print("ERROR: No speech detected.")
            sys.exit(1)
        print(f"  {len(segments)} segments transcribed.")

        print("\n[2/5] Diarizing speakers...")
        import diarizer
        turns = diarizer.diarize(wav_path, hf_token=hf_token, num_speakers=num_speakers)
        speakers = sorted(set(t["speaker"] for t in turns))
        print(f"  {len(turns)} speaker turns, {len(speakers)} speakers: {speakers}")

        print("\n[3/5] Aligning transcript with speakers...")
        import aligner
        aligned = aligner.align(segments, turns)
        fireflies = aligner.format_fireflies_transcript(aligned)
        print("  Done.")

        plain = "\n".join(
            f"{s['speaker']} {s['timestamp_str']}: {s['text']}"
            for s in aligned["sentence_level"]
        )

        print("\n[4/5] Generating summary...")
        import reporter
        try:
            summary = reporter.generate_summary(plain)
        except Exception as e:
            summary = f"[Summary failed: {e}]"

        print("\n[5/5] Generating action plan...")
        try:
            action_plan = reporter.generate_action_plan(plain)
            print("  Done.")
        except Exception as e:
            action_plan = f"[Action plan failed: {e}]"

        print("\nExporting...")
        import exporter
        saved = exporter.export(
            meeting_dt=datetime.now(),
            aligned=aligned,
            fireflies_transcript=fireflies,
            summary=summary,
            action_plan=action_plan,
            formats=formats,
        )

        print(f"\n{'=' * 60}")
        print(f"Saved to: {Path(saved[0]).parent}")
        for f in saved:
            print(f"  {Path(f).name}")
        print(f"{'=' * 60}")

    except subprocess.CalledProcessError as e:
        print(f"ERROR: ffmpeg failed: {e.stderr.decode() if e.stderr else e}")
        sys.exit(1)
    finally:
        if wav_path and wav_path not in [str(Path(f).resolve()) for f in args.files]:
            try:
                Path(wav_path).unlink(missing_ok=True)
            except OSError:
                pass


if __name__ == "__main__":
    main()
