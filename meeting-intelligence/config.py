import os
from pathlib import Path
from dotenv import load_dotenv

# Try .env in project root first, then in the meeting-intelligence directory
env_path = Path(__file__).resolve().parent.parent / ".env"
if not env_path.exists():
    env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(env_path)

# Whisper settings
WHISPER_MODEL = "large-v3"
WHISPER_LANGUAGE = None  # auto-detect (handles Hindi+English code-switching)
WHISPER_COMPUTE_TYPE = "int8"  # CPU-optimized

# Ollama settings
OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_BASE_URL = "http://localhost:11434"

# Output settings
OUTPUT_DIR = "outputs"

# HuggingFace token for pyannote
HF_TOKEN = os.environ.get("HF_TOKEN", "")
