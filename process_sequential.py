"""
Transcribe + Ollama summary CLI (no diarization). Lighter-weight pipeline.

Usage:
  python process_sequential.py audio1.m4a [audio2.m4a ...]  [flags]
  python process_sequential.py recording.mp3 --model mistral
  python process_sequential.py *.m4a --output my_outputs

Flags:
  --output DIR    Output directory (default: "outputs")
  --model NAME    Ollama model to use (default: llama3.1:8b)
"""

import argparse
import json
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


def call_ollama(prompt: str, model: str) -> str:
    import urllib.request

    data = json.dumps({
        "model": model,
        "system": "You are a helpful assistant. Respond concisely.",
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": 4000},
    }).encode()

    req = urllib.request.Request(
        f"{config.OLLAMA_BASE_URL}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())["response"]


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe + Ollama summary (no diarization)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("files", nargs="+", help="Audio file(s) to process")
    parser.add_argument("--output", default="outputs", help="Output directory")
    parser.add_argument("--model", default=config.OLLAMA_MODEL, help="Ollama model name")
    args = parser.parse_args()

    config.OUTPUT_DIR = args.output
    model = args.model

    wav_path = None
    try:
        print("=" * 60)
        print("MEETING INTELLIGENCE — Sequential Processor")
        print("=" * 60)

        wav_path = concat_to_wav(args.files)
        print(f"Audio prepared: {wav_path}")

        print("\n[1/3] Transcribing...")
        import transcriber
        segments = transcriber.transcribe(wav_path)
        if not segments:
            print("ERROR: No speech detected.")
            sys.exit(1)
        print(f"  {len(segments)} segments transcribed.")

        full_text = "\n".join(
            f"[{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['text']}"
            for seg in segments
        )

        plain_text = "\n".join(seg["text"] for seg in segments)

        print(f"\n[2/3] Generating summary via {model}...")
        summary_prompt = (
            "You are a meeting analyst. Given this meeting transcript, write a concise executive summary "
            "covering: main topics discussed, key decisions made, important points raised. "
            "Be specific and use bullet points where appropriate.\n\n"
            f"Meeting transcript:\n{plain_text}"
        )
        try:
            summary = call_ollama(summary_prompt, model)
        except Exception as e:
            summary = f"[Summary failed: {e}]"

        print(f"\n[3/3] Generating action plan via {model}...")
        action_prompt = (
            "Extract all action items from this meeting transcript. "
            "Format as a numbered list with: action item, owner (if mentioned), deadline (if mentioned). "
            "If no clear action items, write 'No specific action items identified.'\n\n"
            f"Meeting transcript:\n{plain_text}"
        )
        try:
            action_plan = call_ollama(action_prompt, model)
            print("  Done.")
        except Exception as e:
            action_plan = f"[Action plan failed: {e}]"

        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        md_path = out_dir / f"meeting_analysis_{ts}.md"
        with open(md_path, "w") as f:
            f.write("# Meeting Analysis\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
            f.write(f"**Model:** {model}\n\n")
            f.write("---\n\n")
            f.write("## Executive Summary\n\n")
            f.write(summary + "\n\n")
            f.write("---\n\n")
            f.write("## Action Items\n\n")
            f.write(action_plan + "\n\n")
            f.write("---\n\n")
            f.write("## Full Transcript\n\n")
            f.write(full_text + "\n")

        print(f"\n{'=' * 60}")
        print(f"Saved: {md_path}")
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
