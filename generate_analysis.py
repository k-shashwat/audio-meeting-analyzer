"""
Generate meeting summary and action items from an existing transcript file.

Usage:
  python generate_analysis.py transcript.txt  [flags]
  python generate_analysis.py transcript.txt --output my_outputs
  python generate_analysis.py transcript.txt --model mistral

Flags:
  --output DIR    Output directory (default: "outputs")
  --model NAME    Ollama model to use (default: llama3.1:8b)
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "meeting-intelligence"))

import config


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
        description="Generate summary and action items from a transcript file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("transcript", help="Path to transcript text file")
    parser.add_argument("--output", default="outputs", help="Output directory")
    parser.add_argument("--model", default=config.OLLAMA_MODEL, help="Ollama model name")
    args = parser.parse_args()

    transcript_path = Path(args.transcript)
    if not transcript_path.exists():
        print(f"ERROR: Transcript file not found: {args.transcript}")
        sys.exit(1)

    transcript = transcript_path.read_text()
    print(f"Transcript loaded: {len(transcript)} chars")
    model = args.model

    print(f"\nGenerating summary via {model}...")
    summary_prompt = (
        "You are a meeting analyst. Given this meeting transcript, write a concise executive summary "
        "covering: main topics discussed, key decisions made, important points raised. "
        "Be specific and use bullet points where appropriate.\n\n"
        f"Meeting transcript:\n{transcript}"
    )
    try:
        summary = call_ollama(summary_prompt, model)
    except Exception as e:
        summary = f"[Summary failed: {e}]"

    print(f"Generating action plan via {model}...")
    action_prompt = (
        "Extract all action items from this meeting transcript. "
        "Format as a numbered list with: action item, owner (if mentioned), deadline (if mentioned). "
        "If no clear action items, write 'No specific action items identified.'\n\n"
        f"Meeting transcript:\n{transcript}"
    )
    try:
        action_plan = call_ollama(action_prompt, model)
        print("Done.")
    except Exception as e:
        action_plan = f"[Action plan failed: {e}]"

    print(f"\n" + "=" * 60)
    print("EXECUTIVE SUMMARY")
    print("=" * 60)
    print(summary)

    print(f"\n" + "=" * 60)
    print("ACTION ITEMS")
    print("=" * 60)
    print(action_plan)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = out_dir / f"meeting_analysis_{ts}.md"

    with open(report_path, "w") as f:
        f.write("# Meeting Analysis\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%B %d, %Y')}\n")
        f.write(f"**Model:** {model}\n\n")
        f.write("---\n\n")
        f.write("## Executive Summary\n\n")
        f.write(summary + "\n\n")
        f.write("---\n\n")
        f.write("## Action Items\n\n")
        f.write(action_plan + "\n")

    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
