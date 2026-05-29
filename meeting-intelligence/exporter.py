"""
Export meeting intelligence outputs in multiple formats.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path

import config

logger = logging.getLogger(__name__)


def _ensure_output_dir(meeting_dt: datetime) -> Path:
    """Create and return the output directory for this meeting."""
    folder_name = meeting_dt.strftime("%Y-%m-%d_%H-%M-%S")
    output_path = Path(config.OUTPUT_DIR) / folder_name
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def _build_markdown(
    meeting_dt: datetime,
    fireflies_transcript: str,
    summary: str,
    action_plan: str,
    sentence_level: list[dict],
) -> str:
    """Build Fireflies-style markdown content."""
    date_str = meeting_dt.strftime("%B %d, %Y at %H:%M:%S")
    lines = [
        "# 🎙️ Meeting Intelligence Report",
        "",
        f"**Date:** {date_str}",
        f"**Total Segments:** {len(sentence_level)}",
        "",
        "---",
        "",
        "## 📋 Executive Summary",
        "",
        summary,
        "",
        "---",
        "",
        "## ✅ Action Plan",
        "",
        action_plan,
        "",
        "---",
        "",
        "## 📝 Full Transcript",
        "",
        fireflies_transcript,
        "",
    ]
    return "\n".join(lines)


def _build_plain_text(
    meeting_dt: datetime,
    fireflies_transcript: str,
    summary: str,
    action_plan: str,
) -> str:
    """Build plain text content."""
    date_str = meeting_dt.strftime("%B %d, %Y at %H:%M:%S")
    # Strip markdown bold markers for plain text
    clean_transcript = fireflies_transcript.replace("**", "")
    lines = [
        "MEETING INTELLIGENCE REPORT",
        "=" * 40,
        f"Date: {date_str}",
        "",
        "EXECUTIVE SUMMARY",
        "-" * 40,
        summary,
        "",
        "ACTION PLAN",
        "-" * 40,
        action_plan,
        "",
        "FULL TRANSCRIPT",
        "-" * 40,
        clean_transcript,
        "",
    ]
    return "\n".join(lines)


def _build_json_data(
    meeting_dt: datetime,
    aligned: dict,
    summary: str,
    action_plan: str,
) -> dict:
    """Build structured JSON data."""
    return {
        "meeting_date": meeting_dt.isoformat(),
        "summary": summary,
        "action_plan": action_plan,
        "transcript": {
            "sentence_level": aligned["sentence_level"],
            "speaker_turns": [
                {
                    "timestamp_start": t["timestamp_start"],
                    "timestamp_end": t["timestamp_end"],
                    "timestamp_str": t["timestamp_str"],
                    "speaker": t["speaker"],
                    "full_text": t["full_text"],
                    "sentences": t["sentences"],
                }
                for t in aligned["speaker_turns"]
            ],
        },
        "metadata": {
            "total_sentences": len(aligned["sentence_level"]),
            "total_speaker_turns": len(aligned["speaker_turns"]),
            "speakers": sorted(
                set(s["speaker"] for s in aligned["sentence_level"])
            ),
        },
    }


def _save_pdf(output_path: Path, meeting_dt: datetime, summary: str, action_plan: str, fireflies_transcript: str) -> Path:
    """Save PDF using reportlab."""
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
    from reportlab.lib import colors

    pdf_path = output_path / "meeting_report.pdf"
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        rightMargin=inch,
        leftMargin=inch,
        topMargin=inch,
        bottomMargin=inch,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Title"],
        fontSize=20,
        spaceAfter=12,
        textColor=colors.HexColor("#1a1a2e"),
    )
    heading_style = ParagraphStyle(
        "CustomHeading",
        parent=styles["Heading2"],
        fontSize=14,
        spaceBefore=16,
        spaceAfter=8,
        textColor=colors.HexColor("#16213e"),
    )
    body_style = ParagraphStyle(
        "CustomBody",
        parent=styles["Normal"],
        fontSize=10,
        spaceAfter=6,
        leading=14,
    )
    mono_style = ParagraphStyle(
        "Mono",
        parent=styles["Normal"],
        fontSize=9,
        fontName="Courier",
        spaceAfter=4,
        leading=13,
        leftIndent=12,
    )

    story = []
    date_str = meeting_dt.strftime("%B %d, %Y at %H:%M:%S")

    story.append(Paragraph("Meeting Intelligence Report", title_style))
    story.append(Paragraph(f"<b>Date:</b> {date_str}", body_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    story.append(Spacer(1, 12))

    # Summary
    story.append(Paragraph("Executive Summary", heading_style))
    for line in summary.split("\n"):
        line = line.strip()
        if line:
            story.append(Paragraph(line, body_style))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))

    # Action Plan
    story.append(Paragraph("Action Plan", heading_style))
    for line in action_plan.split("\n"):
        line = line.strip()
        if line:
            story.append(Paragraph(line, body_style))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))

    # Transcript
    story.append(Paragraph("Full Transcript", heading_style))
    for line in fireflies_transcript.split("\n"):
        clean = line.replace("**", "").strip()
        if not clean:
            story.append(Spacer(1, 6))
            continue
        if line.startswith("**"):
            # Speaker header line
            story.append(Paragraph(f"<b>{clean}</b>", body_style))
        else:
            story.append(Paragraph(clean, mono_style))

    doc.build(story)
    logger.info(f"PDF saved: {pdf_path}")
    return pdf_path


def export(
    meeting_dt: datetime,
    aligned: dict,
    fireflies_transcript: str,
    summary: str,
    action_plan: str,
    formats: list[str],
) -> list[str]:
    """
    Export meeting data in the specified formats.

    Args:
        meeting_dt: Meeting datetime (used for folder name)
        aligned: Output from aligner.align()
        fireflies_transcript: Formatted transcript string
        summary: Generated summary text
        action_plan: Generated action plan text
        formats: List of format strings from ["Markdown", "JSON", "PDF", "Plain Text"]

    Returns:
        List of saved file paths (absolute)
    """
    output_path = _ensure_output_dir(meeting_dt)
    saved_files = []

    format_set = {f.lower() for f in formats}

    if "markdown" in format_set or "md" in format_set:
        content = _build_markdown(
            meeting_dt, fireflies_transcript, summary, action_plan,
            aligned["sentence_level"]
        )
        fpath = output_path / "meeting_report.md"
        fpath.write_text(content, encoding="utf-8")
        saved_files.append(str(fpath.resolve()))
        logger.info(f"Markdown saved: {fpath}")

    if "json" in format_set:
        data = _build_json_data(meeting_dt, aligned, summary, action_plan)
        fpath = output_path / "meeting_report.json"
        fpath.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        saved_files.append(str(fpath.resolve()))
        logger.info(f"JSON saved: {fpath}")

    if "plain text" in format_set or "txt" in format_set or "text" in format_set:
        content = _build_plain_text(
            meeting_dt, fireflies_transcript, summary, action_plan
        )
        fpath = output_path / "meeting_report.txt"
        fpath.write_text(content, encoding="utf-8")
        saved_files.append(str(fpath.resolve()))
        logger.info(f"Plain text saved: {fpath}")

    if "pdf" in format_set:
        try:
            pdf_path = _save_pdf(
                output_path, meeting_dt, summary, action_plan, fireflies_transcript
            )
            saved_files.append(str(pdf_path.resolve()))
        except Exception as e:
            logger.error(f"PDF export failed: {e}")
            raise

    return saved_files
