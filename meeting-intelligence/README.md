# audio-meeting-analyzer

AI-powered meeting transcription, speaker diarization, and analysis — running fully locally.

Upload a meeting recording (m4a, mp3, mpeg) and get back a Fireflies.ai-style timestamped transcript with speaker labels, an executive summary, and a structured action plan — all without sending your audio to any external service.

---

## Features

- **Multilingual transcription** — Hindi, English, and code-switching handled automatically via `faster-whisper` large-v3
- **Speaker diarization** — Identifies who said what using `pyannote.audio` 3.1
- **Fireflies-style transcript** — Timestamps at both sentence and speaker-turn level
- **AI summary + action plan** — Generated locally via Ollama (`llama3.1:8b`)
- **Multiple export formats** — Markdown, JSON, PDF, plain text; chosen at runtime
- **Timestamped output folders** — Saved to `outputs/YYYY-MM-DD_HH-MM-SS/`
- **Gradio web UI** — Clean browser interface, no CLI required

---

## Architecture

```
audio-meeting-analyzer/
├── app.py           # Gradio web UI + processing pipeline orchestration
├── transcriber.py   # faster-whisper transcription
├── diarizer.py      # pyannote.audio speaker diarization
├── aligner.py       # Merge transcript + diarization → timestamped segments
├── reporter.py      # Ollama LLM: summary + action plan
├── exporter.py      # Output format handlers (MD, JSON, PDF, TXT)
├── config.py        # Model names, paths, env vars
├── requirements.txt
└── outputs/         # Created at runtime
```

### Pipeline flow

```
Audio file (m4a/mp3/mpeg)
        │
        ▼
  ffmpeg → WAV (16kHz mono)
        │
        ├──► faster-whisper ──► word-level segments [{start, end, text, words}]
        │
        └──► pyannote.audio ──► speaker turns [{start, end, speaker}]
                │
                ▼
           aligner.py ──► Fireflies-style transcript
                │
                ▼
          Ollama llama3.1:8b
                │
          ┌─────┴─────┐
          ▼           ▼
       Summary    Action Plan
                │
                ▼
          exporter.py ──► outputs/YYYY-MM-DD_HH-MM-SS/
```

### Module descriptions

| Module | Responsibility |
|--------|---------------|
| `transcriber.py` | Loads faster-whisper large-v3 with int8 quantization (CPU-optimized). Returns word-level timestamped segments. |
| `diarizer.py` | Loads pyannote/speaker-diarization-3.1 via HuggingFace. Returns speaker turn intervals. Accepts optional `num_speakers` hint. |
| `aligner.py` | Assigns speaker labels to transcript segments via overlap matching. Produces both sentence-level and speaker-turn-level views. Formats Fireflies-style output. |
| `reporter.py` | Calls Ollama HTTP API with two prompts: executive summary and action item extraction. Handles connection errors gracefully. |
| `exporter.py` | Writes output files in selected formats. PDF uses reportlab. JSON includes full structured data. |
| `app.py` | Gradio Blocks UI. Runs pipeline as a generator to stream live status updates to the browser. |

---

## Prerequisites

### 1. Python 3.9+

```bash
python3 --version
```

### 2. ffmpeg

Required for audio conversion (m4a/mp3/mpeg → WAV).

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Windows — download from https://ffmpeg.org/download.html and add to PATH
```

### 3. Ollama

Download and install from **https://ollama.com/download**, then:

```bash
# Pull the model (one-time, ~4.7 GB download)
ollama pull llama3.1:8b

# Start the server (keep this terminal open while using the app)
ollama serve
```

### 4. HuggingFace token + model access

Speaker diarization requires a free HuggingFace account and two model license acceptances:

1. Create account at **https://huggingface.co**
2. Generate a token at **https://huggingface.co/settings/tokens** (read access is sufficient)
3. Accept the license at **https://huggingface.co/pyannote/speaker-diarization-3.1**
4. Accept the license at **https://huggingface.co/pyannote/segmentation-3.0**

Both acceptances are one-time and free.

---

## Installation

```bash
git clone https://github.com/k-shashwat/audio-meeting-analyzer.git
cd audio-meeting-analyzer

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

> **Note:** `torch` is ~2 GB. Installation takes a few minutes on first run.

---

## Configuration

Set your HuggingFace token using one of these methods:

**Option A — Environment variable (recommended)**
```bash
export HF_TOKEN="hf_your_token_here"
```

**Option B — .env file**

Create a `.env` file in the project root:
```
HF_TOKEN=hf_your_token_here
```

**Option C — UI input**

Enter your token in the web UI's password field each session.

---

## Running

Ensure Ollama is running in a separate terminal (`ollama serve`), then:

```bash
python app.py
```

Open **http://localhost:7860** in your browser.

---

## Usage

1. **Upload** your audio file (m4a, mp3, mpeg, wav supported)
2. **HF Token** — pre-filled if `HF_TOKEN` env var is set, otherwise enter it manually
3. **Number of Speakers** — set to the expected number of participants (improves diarization accuracy; use 1 if unsure)
4. **Output Formats** — select any combination: Markdown, JSON, PDF, Plain Text
5. Click **🚀 Process Meeting**

Live status updates stream as each step completes:

```
⏳ Converting audio to WAV format...
🎙️ Transcribing audio... (this may take a few minutes)
👥 Identifying speakers...
🔗 Aligning transcript with speaker turns...
📝 Transcript ready. Generating summary...
✅ Summary done. Generating action plan...
💾 Saving outputs...
✅ Processing complete!
```

Processing time on CPU: roughly **5–10× real-time** (e.g. a 1-hour meeting takes 5–10 minutes).

---

## Output Format

All files saved to `outputs/YYYY-MM-DD_HH-MM-SS/`.

### Fireflies-style Markdown transcript

```markdown
## 📝 Full Transcript

**[00:00:05] Speaker 1**
Hello everyone, welcome to the meeting.

[00:00:08] Thanks for joining today.

**[00:00:12] Speaker 2**
Thank you for having us. Let's get started.
```

- **Bold lines** = new speaker turn with timestamp
- Plain `[HH:MM:SS]` lines = sentence-level timestamps within the same speaker turn

### JSON structure

```json
{
  "meeting_date": "2024-01-15T14:30:00",
  "summary": "Executive summary text...",
  "action_plan": "1. Action item...\n2. ...",
  "transcript": {
    "sentence_level": [
      {
        "timestamp_start": 5.2,
        "timestamp_end": 8.1,
        "timestamp_str": "[00:00:05]",
        "speaker": "Speaker 1",
        "text": "Hello everyone, welcome to the meeting."
      }
    ],
    "speaker_turns": [...]
  }
}
```

---

## Configuration Reference

Edit `config.py` to change defaults without touching the UI:

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_MODEL` | `"large-v3"` | Whisper model size. Options: `tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3` |
| `WHISPER_LANGUAGE` | `None` | Force a language (`"hi"`, `"en"`) or `None` for auto-detect |
| `WHISPER_COMPUTE_TYPE` | `"int8"` | `"int8"` for CPU. `"float16"` for GPU. |
| `OLLAMA_MODEL` | `"llama3.1:8b"` | Any model you have pulled in Ollama |
| `OLLAMA_BASE_URL` | `"http://localhost:11434"` | Ollama server URL |
| `OUTPUT_DIR` | `"outputs"` | Root folder for saved output files |

### Using a smaller/faster Whisper model

For faster results at the cost of accuracy, change `config.py`:

```python
WHISPER_MODEL = "medium"   # ~2x faster than large-v3
WHISPER_MODEL = "small"    # ~4x faster, lower accuracy
```

### Switching the LLM

Pull any Ollama model and update `config.py`:

```bash
ollama pull mistral
```

```python
OLLAMA_MODEL = "mistral"
```

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `ffmpeg: command not found` | Install ffmpeg (see Prerequisites) |
| `401 Unauthorized` from HuggingFace | Accept both model licenses (links in Prerequisites §4) |
| `Connection refused` on Ollama | Run `ollama serve` in a separate terminal |
| Port 7860 already in use | Run `kill $(lsof -ti:7860)` then retry |
| Very slow transcription | Set `WHISPER_MODEL = "medium"` in `config.py` |
| `No speech detected` | Verify the audio file plays correctly and isn't silent |
| Large-v3 model download slow | First run only — model is cached after initial download |

---

## Requirements

```
faster-whisper>=1.0.0
pyannote.audio>=3.1.0
torch>=2.0.0
torchaudio>=2.0.0
gradio==3.50.2
requests
reportlab>=4.0.0
python-dotenv
numpy
pydub
```

> Gradio is pinned to `3.50.2` for Python 3.9 compatibility.
