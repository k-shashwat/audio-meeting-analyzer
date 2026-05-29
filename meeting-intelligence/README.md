# Meeting Intelligence

AI-powered meeting transcription, speaker diarization, and analysis — running entirely locally.

## Features

- **Transcription** — faster-whisper large-v3 with word-level timestamps; auto-detects language including Hindi+English code-switching
- **Speaker diarization** — pyannote.audio 3.1 identifies who spoke when
- **Fireflies-style transcript** — timestamped per sentence and per speaker turn
- **AI analysis** — Ollama llama3.1:8b generates executive summary and action plan
- **Multiple output formats** — Markdown, JSON, PDF, plain text
- **Gradio web UI** — simple browser-based interface

## Requirements

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/) installed and on PATH (for audio conversion)
- [Ollama](https://ollama.ai/) running locally with llama3.1:8b model
- HuggingFace account with access to pyannote models

## Setup

### 1. Install Python dependencies

```bash
cd meeting-intelligence
pip install -r requirements.txt
```

### 2. Install ffmpeg

**Ubuntu/Debian:**
```bash
sudo apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows:** Download from https://ffmpeg.org/download.html and add to PATH.

### 3. Set up HuggingFace Token

You need a HuggingFace token to download the pyannote diarization model.

1. Create an account at [huggingface.co](https://huggingface.co)
2. Go to [Settings → Tokens](https://huggingface.co/settings/tokens) and create a read token
3. Accept the model conditions at:
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0

Set the token as an environment variable:
```bash
export HF_TOKEN="hf_your_token_here"
```

Or create a `.env` file in the project directory:
```
HF_TOKEN=hf_your_token_here
```

You can also enter the token directly in the web UI.

### 4. Set up Ollama

1. Install Ollama from [ollama.ai](https://ollama.ai)
2. Start the Ollama server:
   ```bash
   ollama serve
   ```
3. Pull the llama3.1:8b model:
   ```bash
   ollama pull llama3.1:8b
   ```

## Running

```bash
python app.py
```

Then open your browser at **http://localhost:7860**

## Usage

1. Upload an audio file (m4a, mp3, mpeg, wav)
2. Enter your HuggingFace token (or set `HF_TOKEN` env var)
3. Set the number of speakers (optional hint for diarization)
4. Select output formats (Markdown, JSON, PDF, Plain Text)
5. Click **Process Meeting**

Processing a 1-hour meeting on CPU typically takes 5–15 minutes.

## Output

Files are saved to `outputs/YYYY-MM-DD_HH-MM-SS/`:

| File | Description |
|------|-------------|
| `meeting_report.md` | Fireflies-style markdown with summary, action plan, transcript |
| `meeting_report.json` | Structured JSON with all data |
| `meeting_report.txt` | Plain text version |
| `meeting_report.pdf` | Formatted PDF report |

### Transcript Format (Markdown)

```
## 📝 Full Transcript

**[00:00:05] Speaker 1**
Hello everyone, welcome to the meeting.

[00:00:08] Thanks for joining today.

**[00:00:12] Speaker 2**
Thank you for having us. Let's get started.
```

- **Bold lines** = new speaker turn with timestamp
- Plain bracket lines = sentence-level timestamps within the same speaker turn

## Project Structure

```
meeting-intelligence/
├── app.py            # Gradio UI entry point
├── transcriber.py    # faster-whisper transcription
├── diarizer.py       # pyannote.audio speaker diarization
├── aligner.py        # merge transcript + diarization
├── reporter.py       # Ollama LLM calls for summary + action plan
├── exporter.py       # Output format handlers (md, json, pdf, txt)
├── config.py         # Configuration constants
├── requirements.txt
└── outputs/          # Generated meeting reports
```

## Configuration

Edit `config.py` to change defaults:

- `WHISPER_MODEL` — Whisper model size (default: `large-v3`)
- `OLLAMA_MODEL` — Ollama model (default: `llama3.1:8b`)
- `OLLAMA_BASE_URL` — Ollama server URL (default: `http://localhost:11434`)
- `OUTPUT_DIR` — Output directory (default: `outputs`)

## Notes

- The whisper large-v3 model (~3GB) is downloaded on first run and cached in `~/.cache/huggingface/`
- The pyannote model (~200MB) is also downloaded on first run
- For faster processing on CPU, you can change `WHISPER_MODEL` to `medium` or `small` in `config.py`
