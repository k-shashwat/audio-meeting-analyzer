"""
Speaker diarization using pyannote.audio.
Returns speaker turn segments with start/end times.
"""

import logging
from pyannote.audio import Pipeline
import torch
import config

logger = logging.getLogger(__name__)


def diarize(audio_path: str, hf_token: str, num_speakers: int = None) -> list[dict]:
    """
    Perform speaker diarization on an audio file.

    Args:
        audio_path: Path to audio file (wav format required)
        hf_token: HuggingFace access token
        num_speakers: Optional hint for number of speakers

    Returns:
        List of diarization turn dicts:
        {
            "start": float,
            "end": float,
            "speaker": str  (e.g. "SPEAKER_00")
        }
    """
    if not hf_token:
        raise ValueError(
            "HF_TOKEN is required for speaker diarization. "
            "Get your token at https://huggingface.co/settings/tokens "
            "and accept conditions at https://huggingface.co/pyannote/speaker-diarization-3.1"
        )

    logger.info("Loading pyannote speaker diarization pipeline")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )

    # Use CPU
    pipeline.to(torch.device("cpu"))

    logger.info(f"Diarizing {audio_path} (num_speakers={num_speakers})")

    kwargs = {}
    if num_speakers and num_speakers > 0:
        kwargs["num_speakers"] = num_speakers

    diarization = pipeline(audio_path, **kwargs)

    turns = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        turns.append(
            {
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker,
            }
        )

    # Sort by start time
    turns.sort(key=lambda x: x["start"])

    logger.info(f"Diarization complete: {len(turns)} turns")
    speakers = sorted(set(t["speaker"] for t in turns))
    logger.info(f"Speakers found: {speakers}")

    return turns
