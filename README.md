# Meeting Intelligence

AI-powered meeting transcription, speaker diarization, and analysis — running fully locally on your machine.

Upload a meeting recording and get back a Fireflies.ai-style timestamped transcript with speaker labels, an executive summary, and a structured action plan. Nothing leaves your computer.

---

## Features

- **Multilingual transcription** — Hindi, English, and code-switching handled automatically via `faster-whisper` large-v3
- **Speaker diarization** — Identifies who said what using `pyannote.audio` 3.1
- **Fireflies-style transcript** — Timestamps at both sentence and speaker-turn level
- **AI summary + action plan** — Generated locally via Ollama (`llama3.1:8b` or any model you choose)
- **Multiple export formats** — Markdown, JSON, PDF, plain text
- **Web UI + CLI** — Gradio browser interface or command-line scripts
- **LAN/mobile access** — Access the web UI from your phone on the same WiFi

---

## Quick Start

### Prerequisites

- **Python 3.9+**
- **ffmpeg** — [Download](https://ffmpeg.org/download.html) or `brew install ffmpeg` (macOS) / `apt install ffmpeg` (Linux)
- **Ollama** — [Download](https://ollama.com/download), then `ollama pull llama3.1:8b` and `ollama serve`
- **HuggingFace token** — Free at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). Accept model licenses at [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) and [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)

### Installation

```bash
git clone https://github.com/k-shashwat/audio-meeting-analyzer.git
cd audio-meeting-analyzer

python3 -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

cd meeting-intelligence
pip install -r requirements.txt
cd ..
```

### Configuration

Copy the example env file and add your HuggingFace token:

```bash
cp .env.example .env
# Edit .env and replace hf_your_token_here with your actual token
```

### Run

**Web UI (recommended):**
```bash
cd meeting-intelligence
python app.py
# Open http://localhost:7860
# From mobile: http://<your-laptop-ip>:7860
```

**CLI — Full pipeline (transcribe + diarize + analyze):**
```bash
python process_meeting.py meeting.m4a --speakers 3 --formats md json pdf
```

**CLI — Quick transcription + summary (no diarization):**
```bash
python process_sequential.py recording.mp3
```

**CLI — Analyze an existing transcript:**
```bash
python generate_analysis.py transcript.txt
```

**CLI — Transcribe audio to text only:**
```bash
python transcribe_combined.py --input audio.m4a --output transcript.txt --model medium
```

---

## Usage

### Web UI

1. Open `http://localhost:7860` in your browser
2. Upload your audio file (m4a, mp3, mpeg, wav, mp4)
3. Enter your HuggingFace token (or set `HF_TOKEN` in `.env`)
4. Adjust the speaker count hint for better diarization
5. Select output formats
6. Click **Process Meeting**

### CLI

```
python process_meeting.py --help
python process_sequential.py --help
python generate_analysis.py --help
python transcribe_combined.py --help
```

---

## Configuration

Edit `meeting-intelligence/config.py` to change defaults:

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_MODEL` | `large-v3` | Whisper model: `tiny`, `base`, `small`, `medium`, `large-v3` |
| `WHISPER_LANGUAGE` | `None` | Force language (`"en"`, `"hi"`) or `None` for auto-detect |
| `WHISPER_COMPUTE_TYPE` | `int8` | `int8` for CPU, `float16` for GPU |
| `OLLAMA_MODEL` | `llama3.1:8b` | Any model pulled in Ollama |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server address |
| `OUTPUT_DIR` | `outputs` | Where reports are saved |

---

## Architecture

```
meeting-intelligence/
├── app.py           # Gradio web UI + pipeline orchestration
├── transcriber.py   # faster-whisper transcription
├── diarizer.py      # pyannote.audio speaker diarization
├── aligner.py       # Merge transcript + diarization -> speaker-labeled segments
├── reporter.py      # Ollama LLM: summary + action plan generation
├── exporter.py      # Output formats: Markdown, JSON, PDF, plain text
├── config.py        # Central configuration
└── requirements.txt
```

**Pipeline flow:** Audio → ffmpeg (16kHz mono WAV) → faster-whisper (transcript) + pyannote (speakers) → aligner (merge) → Ollama (summary + actions) → exporter (save)

---

## Output

All results are saved to `outputs/YYYY-MM-DD_HH-MM-SS/` containing:

- `meeting_report.md` — Fireflies-style speaker-labeled transcript with summary and action plan
- `meeting_report.json` — Full structured data with metadata
- `meeting_report.pdf` — Formatted PDF report
- `meeting_report.txt` — Plain text version

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `ffmpeg: command not found` | Install ffmpeg ([download](https://ffmpeg.org/download.html)) |
| `401 Unauthorized` from HuggingFace | Accept model licenses at the links above |
| `Connection refused` on Ollama | Run `ollama serve` in a separate terminal |
| Port 7860 already in use | Run `kill $(lsof -ti:7860)` then retry |
| Slow transcription | Set `WHISPER_MODEL = "medium"` in `config.py` |
| `No speech detected` | Verify the audio plays correctly and isn't silent |
| Can't access from mobile | Ensure laptop and phone are on the same WiFi. Try `http://<your-ip>:7860` |

---

## License

MIT
