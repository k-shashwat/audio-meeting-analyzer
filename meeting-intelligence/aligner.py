"""
Align whisper transcript segments with pyannote speaker diarization turns.
Produces timestamped, speaker-labeled segments.
"""

import logging
from datetime import timedelta

logger = logging.getLogger(__name__)


def _format_timestamp(seconds: float) -> str:
    """Format seconds as [HH:MM:SS]."""
    td = timedelta(seconds=int(seconds))
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"[{hours:02d}:{minutes:02d}:{secs:02d}]"


def _find_speaker_for_segment(
    seg_start: float, seg_end: float, diarization_turns: list[dict]
) -> str:
    """
    Find the most overlapping speaker for a given time segment.
    Falls back to 'Unknown' if no overlap found.
    """
    seg_mid = (seg_start + seg_end) / 2.0
    best_speaker = "Unknown"
    best_overlap = 0.0

    for turn in diarization_turns:
        overlap_start = max(seg_start, turn["start"])
        overlap_end = min(seg_end, turn["end"])
        overlap = max(0.0, overlap_end - overlap_start)

        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = turn["speaker"]

    # Also try midpoint match if no overlap found
    if best_speaker == "Unknown":
        for turn in diarization_turns:
            if turn["start"] <= seg_mid <= turn["end"]:
                best_speaker = turn["speaker"]
                break

    return best_speaker


def _normalize_speaker_labels(segments: list[dict]) -> list[dict]:
    """
    Rename SPEAKER_00 -> Speaker 1, SPEAKER_01 -> Speaker 2, etc.
    """
    speaker_map = {}
    counter = 1
    for seg in segments:
        raw = seg["speaker"]
        if raw not in speaker_map and raw != "Unknown":
            speaker_map[raw] = f"Speaker {counter}"
            counter += 1
    if "Unknown" in [s["speaker"] for s in segments]:
        speaker_map["Unknown"] = "Unknown"

    result = []
    for seg in segments:
        new_seg = dict(seg)
        new_seg["speaker"] = speaker_map.get(seg["speaker"], seg["speaker"])
        result.append(new_seg)
    return result


def align(
    transcript_segments: list[dict],
    diarization_turns: list[dict],
) -> dict:
    """
    Merge transcript segments with speaker diarization.

    Args:
        transcript_segments: Output from transcriber.transcribe()
        diarization_turns: Output from diarizer.diarize()

    Returns:
        {
            "sentence_level": [
                {
                    "timestamp_start": float,
                    "timestamp_end": float,
                    "timestamp_str": "[HH:MM:SS]",
                    "speaker": str,
                    "text": str,
                }
            ],
            "speaker_turns": [
                {
                    "timestamp_start": float,
                    "timestamp_end": float,
                    "timestamp_str": "[HH:MM:SS]",
                    "speaker": str,
                    "sentences": [
                        {
                            "timestamp_str": "[HH:MM:SS]",
                            "text": str,
                        }
                    ],
                    "full_text": str,
                }
            ]
        }
    """
    # Step 1: assign speaker to each transcript segment
    sentence_level = []
    for seg in transcript_segments:
        if not seg["text"].strip():
            continue
        speaker = _find_speaker_for_segment(
            seg["start"], seg["end"], diarization_turns
        )
        sentence_level.append(
            {
                "timestamp_start": seg["start"],
                "timestamp_end": seg["end"],
                "timestamp_str": _format_timestamp(seg["start"]),
                "speaker": speaker,
                "text": seg["text"].strip(),
            }
        )

    # Normalize speaker labels
    sentence_level = _normalize_speaker_labels(sentence_level)

    # Step 2: merge consecutive segments from same speaker into turns
    speaker_turns = []
    if sentence_level:
        current_turn = {
            "timestamp_start": sentence_level[0]["timestamp_start"],
            "timestamp_end": sentence_level[0]["timestamp_end"],
            "timestamp_str": sentence_level[0]["timestamp_str"],
            "speaker": sentence_level[0]["speaker"],
            "sentences": [
                {
                    "timestamp_str": sentence_level[0]["timestamp_str"],
                    "text": sentence_level[0]["text"],
                }
            ],
        }

        for seg in sentence_level[1:]:
            if seg["speaker"] == current_turn["speaker"]:
                # Continue same speaker turn
                current_turn["timestamp_end"] = seg["timestamp_end"]
                current_turn["sentences"].append(
                    {
                        "timestamp_str": seg["timestamp_str"],
                        "text": seg["text"],
                    }
                )
            else:
                # New speaker turn
                current_turn["full_text"] = " ".join(
                    s["text"] for s in current_turn["sentences"]
                )
                speaker_turns.append(current_turn)
                current_turn = {
                    "timestamp_start": seg["timestamp_start"],
                    "timestamp_end": seg["timestamp_end"],
                    "timestamp_str": seg["timestamp_str"],
                    "speaker": seg["speaker"],
                    "sentences": [
                        {
                            "timestamp_str": seg["timestamp_str"],
                            "text": seg["text"],
                        }
                    ],
                }

        # Don't forget the last turn
        current_turn["full_text"] = " ".join(
            s["text"] for s in current_turn["sentences"]
        )
        speaker_turns.append(current_turn)

    logger.info(
        f"Alignment complete: {len(sentence_level)} sentences, {len(speaker_turns)} speaker turns"
    )

    return {
        "sentence_level": sentence_level,
        "speaker_turns": speaker_turns,
    }


def format_fireflies_transcript(aligned: dict) -> str:
    """
    Format aligned segments as a Fireflies.ai-style transcript string.

    Bold lines = new speaker turn with timestamp
    Plain bracket lines = sentence-level timestamps within same speaker turn
    """
    lines = []
    for turn in aligned["speaker_turns"]:
        # First sentence of each speaker turn is bold with speaker label
        sentences = turn["sentences"]
        if not sentences:
            continue

        # Speaker header line (bold)
        lines.append(f"**{turn['timestamp_str']} {turn['speaker']}**")
        lines.append(sentences[0]["text"])

        # Subsequent sentences in same turn
        for sent in sentences[1:]:
            lines.append(f"{sent['timestamp_str']} {sent['text']}")

        lines.append("")  # blank line between turns

    return "\n".join(lines).strip()
