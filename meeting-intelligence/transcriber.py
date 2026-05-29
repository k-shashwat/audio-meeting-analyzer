"""
Transcription using faster-whisper.
Returns segments with word-level timestamps.
"""

from faster_whisper import WhisperModel
import config
import logging

logger = logging.getLogger(__name__)


def transcribe(audio_path: str, language: str = None) -> list[dict]:
    """
    Transcribe audio file using faster-whisper.

    Args:
        audio_path: Path to audio file (wav preferred)
        language: Language code (None for auto-detect)

    Returns:
        List of segment dicts:
        {
            "start": float,
            "end": float,
            "text": str,
            "words": [{"start": float, "end": float, "word": str}, ...]
        }
    """
    logger.info(f"Loading Whisper model: {config.WHISPER_MODEL}")
    model = WhisperModel(
        config.WHISPER_MODEL,
        device="cpu",
        compute_type=config.WHISPER_COMPUTE_TYPE,
    )

    lang = language or config.WHISPER_LANGUAGE
    logger.info(f"Transcribing {audio_path} (language={lang or 'auto'})")

    segments_iter, info = model.transcribe(
        audio_path,
        language=lang,
        word_timestamps=True,
        beam_size=5,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )

    logger.info(
        f"Detected language: {info.language} (probability={info.language_probability:.2f})"
    )

    segments = []
    for seg in segments_iter:
        words = []
        if seg.words:
            for w in seg.words:
                words.append(
                    {
                        "start": w.start,
                        "end": w.end,
                        "word": w.word,
                    }
                )
        segments.append(
            {
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip(),
                "words": words,
            }
        )

    logger.info(f"Transcription complete: {len(segments)} segments")
    return segments
