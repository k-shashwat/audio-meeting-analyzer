"""
Meeting Intelligence - Gradio Web UI
Transcribe, diarize, and analyze meeting audio files.
"""

import logging
import os
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import gradio as gr

import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _convert_to_wav(audio_path: str) -> str:
    """
    Convert audio file to WAV format using ffmpeg.
    Returns path to converted WAV file (in a temp dir).
    """
    suffix = Path(audio_path).suffix.lower()
    if suffix == ".wav":
        return audio_path

    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_wav.close()

    cmd = [
        "ffmpeg", "-y",
        "-i", audio_path,
        "-ar", "16000",   # 16kHz sample rate (optimal for Whisper/pyannote)
        "-ac", "1",       # mono
        "-f", "wav",
        tmp_wav.name,
    ]
    logger.info(f"Converting audio to WAV: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg is not installed or not found in PATH. "
            "Install ffmpeg: brew install ffmpeg (macOS) or apt install ffmpeg (Linux)"
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg conversion failed:\n{result.stderr}"
        )
    return tmp_wav.name


def process_meeting(
    audio_file,
    hf_token_input: str,
    output_formats: list,
    num_speakers: int,
):
    """
    Main processing pipeline.
    Yields (transcript, summary, action_plan, files, status) tuples for live updates.
    """
    # --- Validate inputs ---
    if audio_file is None:
        yield "", "", "", [], "❌ Please upload an audio file."
        return

    # Resolve HF token: prefer UI input, fall back to env
    hf_token = (hf_token_input or "").strip() or config.HF_TOKEN
    if not hf_token:
        yield (
            "", "", "", [],
            "❌ HuggingFace token is required for speaker diarization.\n"
            "Enter your token in the HF Token field, or set the HF_TOKEN environment variable.\n"
            "Get your token at: https://huggingface.co/settings/tokens\n"
            "Accept model conditions at: https://huggingface.co/pyannote/speaker-diarization-3.1"
        )
        return

    if not output_formats:
        output_formats = ["Markdown", "JSON", "PDF", "Plain Text"]

    # Resolve audio path: Gradio 3.x type="file" returns a filepath string,
    # but some versions/backends return an object with .name attribute
    if isinstance(audio_file, str):
        audio_path = audio_file
    elif hasattr(audio_file, "name"):
        audio_path = audio_file.name
    else:
        audio_path = str(audio_file)
    wav_path = None

    try:
        # Step 1: Convert audio
        yield "", "", "", [], "⏳ Converting audio to WAV format..."
        try:
            wav_path = _convert_to_wav(audio_path)
        except RuntimeError as e:
            yield "", "", "", [], f"❌ Audio conversion failed:\n{e}"
            return

        # Step 2: Transcribe
        yield "", "", "", [], "🎙️ Transcribing audio... (this may take a few minutes)"
        try:
            import transcriber
            transcript_segments = transcriber.transcribe(wav_path)
        except Exception as e:
            yield "", "", "", [], f"❌ Transcription failed:\n{e}"
            return

        if not transcript_segments:
            yield "", "", "", [], "❌ No speech detected in the audio file."
            return

        # Step 3: Diarize
        yield "", "", "", [], "👥 Identifying speakers... (this may take a few minutes)"
        try:
            import diarizer
            diarization_turns = diarizer.diarize(
                wav_path,
                hf_token=hf_token,
                num_speakers=int(num_speakers) if num_speakers > 0 else None,
            )
        except ValueError as e:
            yield "", "", "", [], f"❌ Diarization error:\n{e}"
            return
        except Exception as e:
            yield "", "", "", [], f"❌ Speaker diarization failed:\n{e}"
            return

        # Step 4: Align
        yield "", "", "", [], "🔗 Aligning transcript with speaker turns..."
        try:
            import aligner
            aligned = aligner.align(transcript_segments, diarization_turns)
            fireflies_transcript = aligner.format_fireflies_transcript(aligned)
        except Exception as e:
            yield "", "", "", [], f"❌ Alignment failed:\n{e}"
            return

        yield fireflies_transcript, "", "", [], "📝 Transcript ready. Generating summary..."

        # Build plain text transcript for LLM
        plain_transcript = "\n".join(
            f"{seg['speaker']} {seg['timestamp_str']}: {seg['text']}"
            for seg in aligned["sentence_level"]
        )

        # Step 5: Generate summary
        try:
            import reporter
            summary = reporter.generate_summary(plain_transcript)
        except ConnectionError as e:
            summary = f"[Ollama not available: {e}]"
        except Exception as e:
            summary = f"[Summary generation failed: {e}]"

        yield fireflies_transcript, summary, "", [], "✅ Summary done. Generating action plan..."

        # Step 6: Generate action plan
        try:
            action_plan = reporter.generate_action_plan(plain_transcript)
        except ConnectionError as e:
            action_plan = f"[Ollama not available: {e}]"
        except Exception as e:
            action_plan = f"[Action plan generation failed: {e}]"

        yield fireflies_transcript, summary, action_plan, [], "💾 Saving outputs..."

        # Step 7: Export files
        meeting_dt = datetime.now()
        try:
            import exporter
            saved_files = exporter.export(
                meeting_dt=meeting_dt,
                aligned=aligned,
                fireflies_transcript=fireflies_transcript,
                summary=summary,
                action_plan=action_plan,
                formats=output_formats,
            )
        except Exception as e:
            logger.error(f"Export error: {e}", exc_info=True)
            yield (
                fireflies_transcript, summary, action_plan, [],
                f"⚠️ Processing complete but export failed:\n{e}"
            )
            return

        folder = Path(saved_files[0]).parent if saved_files else "outputs/"
        status = (
            f"✅ Processing complete!\n"
            f"📁 Saved {len(saved_files)} file(s) to: {folder}\n"
            f"Files: {', '.join(Path(f).name for f in saved_files)}"
        )
        yield fireflies_transcript, summary, action_plan, saved_files, status

    finally:
        # Cleanup temp WAV if we created one
        if wav_path and wav_path != audio_path:
            try:
                os.unlink(wav_path)
            except OSError:
                pass


# --- Gradio UI ---

with gr.Blocks(
    title="Meeting Intelligence",
    theme=gr.themes.Soft(),
    css="""
        .gradio-container { max-width: 1200px; margin: auto; }
        #status-box textarea { font-family: monospace; }
    """,
) as demo:

    gr.Markdown(
        """
        # 🎙️ Meeting Intelligence
        **AI-powered meeting transcription, speaker diarization, and analysis**

        Upload a meeting recording to get a Fireflies.ai-style transcript with speaker labels,
        executive summary, and action plan — all running locally.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Configuration")

            audio_input = gr.File(
                label="Upload Audio File",
                file_types=[".m4a", ".mp3", ".mpeg", ".wav", ".mp4"],
                type="file",
            )

            hf_token_input = gr.Textbox(
                label="HuggingFace Token",
                placeholder="hf_... (or set HF_TOKEN env var)",
                type="password",
                value=config.HF_TOKEN,
                info="Required for speaker diarization. Get at huggingface.co/settings/tokens",
            )

            num_speakers_input = gr.Slider(
                label="Number of Speakers (hint)",
                minimum=1,
                maximum=10,
                step=1,
                value=2,
                info="Optional hint for diarization. Set to 1 if unsure.",
            )

            format_checkboxes = gr.CheckboxGroup(
                choices=["Markdown", "JSON", "PDF", "Plain Text"],
                value=["Markdown", "JSON", "PDF", "Plain Text"],
                label="Output Formats",
                info="Select one or more formats to save",
            )

            process_btn = gr.Button(
                "🚀 Process Meeting",
                variant="primary",
                size="lg",
            )

            status_output = gr.Textbox(
                label="Status",
                lines=5,
                interactive=False,
                elem_id="status-box",
                placeholder="Status updates will appear here...",
            )

        with gr.Column(scale=2):
            gr.Markdown("### 📊 Results")

            with gr.Tabs():
                with gr.Tab("📝 Transcript"):
                    transcript_output = gr.Textbox(
                        label="Fireflies-style Transcript",
                        lines=20,
                        interactive=False,
                        placeholder="Transcript with speaker labels and timestamps will appear here...",
                    )

                with gr.Tab("📋 Summary"):
                    summary_output = gr.Textbox(
                        label="Executive Summary",
                        lines=10,
                        interactive=False,
                        placeholder="AI-generated summary will appear here...",
                    )

                with gr.Tab("✅ Action Plan"):
                    action_plan_output = gr.Textbox(
                        label="Action Plan",
                        lines=10,
                        interactive=False,
                        placeholder="Extracted action items will appear here...",
                    )

                with gr.Tab("💾 Downloads"):
                    files_output = gr.File(
                        label="Download Output Files",
                        interactive=False,
                        file_count="multiple",
                    )

    gr.Markdown(
        """
        ---
        ### ℹ️ How it works
        1. **Transcription** — [faster-whisper](https://github.com/SYSTRAN/faster-whisper) large-v3 (CPU, int8)
        2. **Diarization** — [pyannote.audio](https://github.com/pyannote/pyannote-audio) 3.1 (requires HF token)
        3. **Analysis** — [Ollama](https://ollama.ai) llama3.1:8b (must be running locally)

        > **Note:** Processing a 1-hour meeting on CPU typically takes 5–15 minutes.
        """
    )

    process_btn.click(
        fn=process_meeting,
        inputs=[audio_input, hf_token_input, format_checkboxes, num_speakers_input],
        outputs=[transcript_output, summary_output, action_plan_output, files_output, status_output],
        show_progress=True,
    )


if __name__ == "__main__":
    demo.queue()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
    )
