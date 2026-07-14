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

    detected_lang = getattr(info, "language", None) or "unknown"
    detected_prob = getattr(info, "language_probability", None) or 0.0
    logger.info(
        f"Detected language: {detected_lang} (probability={detected_prob:.2f})"
    )

    segments = []
    segment_count = 0
    for seg in segments_iter:
        segment_count += 1
        text = seg.text.strip()
        logger.info(f"[{seg.start:.1f}s - {seg.end:.1f}s] {text}")
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
                "text": text,
                "words": words,
            }
        )

    logger.info(f"Transcription complete: {len(segments)} segments")

    # Free model memory
    import gc
    del model
    gc.collect()

    return segments
