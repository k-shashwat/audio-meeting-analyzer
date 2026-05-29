import os
from dotenv import load_dotenv

load_dotenv()

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
