"""
Generate meeting summary and action plan using Ollama LLM.
"""

import logging
import requests
import config

logger = logging.getLogger(__name__)


def _call_ollama(prompt: str, model: str = None, base_url: str = None) -> str:
    """
    Call Ollama API to generate text.

    Args:
        prompt: The full prompt to send
        model: Model name (defaults to config.OLLAMA_MODEL)
        base_url: Ollama base URL (defaults to config.OLLAMA_BASE_URL)

    Returns:
        Generated text string

    Raises:
        ConnectionError: If Ollama is not running
        RuntimeError: On API errors
    """
    model = model or config.OLLAMA_MODEL
    base_url = base_url or config.OLLAMA_BASE_URL
    url = f"{base_url}/api/generate"

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "top_p": 0.9,
        },
    }

    try:
        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            f"Cannot connect to Ollama at {base_url}. "
            "Please ensure Ollama is running: run `ollama serve` in a terminal, "
            f"then pull the model: `ollama pull {model}`"
        )
    except requests.exceptions.Timeout:
        raise RuntimeError(
            f"Ollama request timed out after 300 seconds. "
            "The model may be taking too long to respond."
        )
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"Ollama API error: {e}")

    data = response.json()
    return data.get("response", "").strip()


def generate_summary(transcript_text: str) -> str:
    """
    Generate an executive summary of the meeting.

    Args:
        transcript_text: Full transcript as plain text

    Returns:
        Executive summary string
    """
    prompt = (
        "You are a meeting analyst. Given this meeting transcript, write a concise executive summary "
        "covering: main topics discussed, key decisions made, important points raised. "
        "Be specific and use bullet points where appropriate. Keep it professional and concise.\n\n"
        f"Meeting transcript:\n{transcript_text}"
    )
    logger.info("Generating meeting summary via Ollama")
    return _call_ollama(prompt)


def generate_action_plan(transcript_text: str) -> str:
    """
    Extract action items from the meeting transcript.

    Args:
        transcript_text: Full transcript as plain text

    Returns:
        Numbered action plan string
    """
    prompt = (
        "You are a meeting analyst. Extract all action items from this meeting transcript. "
        "Format as a numbered list with: action item, owner (if mentioned), deadline (if mentioned). "
        "If no clear action items are found, write 'No specific action items identified.' "
        "Be specific and include context from the meeting.\n\n"
        f"Meeting transcript:\n{transcript_text}"
    )
    logger.info("Generating action plan via Ollama")
    return _call_ollama(prompt)


def generate_report(transcript_text: str) -> tuple[str, str]:
    """
    Generate both summary and action plan.

    Args:
        transcript_text: Full transcript as plain text

    Returns:
        Tuple of (summary, action_plan)
    """
    summary = generate_summary(transcript_text)
    action_plan = generate_action_plan(transcript_text)
    return summary, action_plan
