"""LLM language layer and structured summary generator for the doctor.

Converts confirmed NSL signs into a concise Nepali sentence and validated
JSON record for the consulting physician, with strict validation and deterministic fallback.
"""

import json
import os
import re
import time
from typing import Any, Dict, Optional
from dotenv import load_dotenv

from nsl.config import SIGN_NE

# Load environment variables without reading/printing .env directly
load_dotenv()

# Allowed keys in the structured JSON output
ALLOWED_JSON_KEYS = {
    "sentence_ne",
    "complaint",
    "duration_days",
    "allergy",
    "medicine_taken",
}


class ValidationError(Exception):
    """Raised when model response violates format constraints or contradicts confirmed signs."""
    pass


def build_prompt(answers: Dict[str, str]) -> str:
    """Construct a strict prompt instructing Gemma/Gemini to translate signs to Nepali.

    Args:
        answers: Dictionary of confirmed sign answers.

    Returns:
        Structured prompt string.
    """
    lines = ["Confirmed patient signs from intake:"]
    for q_id, sign in answers.items():
        nepali_word = SIGN_NE.get(sign, "")
        lines.append(f"- {q_id}: {sign} (Nepali: {nepali_word})")

    answers_summary = "\n".join(lines)

    return f"""You are a translation assistant in a medical clinic.
Convert the patient's confirmed sign language responses into a single, concise Nepali sentence for the doctor and a structured JSON object.

{answers_summary}

STRICT CLINICAL SAFETY RULES:
1. Use ONLY the provided signs and their Nepali translations. Add NO other symptoms.
2. NEVER diagnose a disease, suggest causes, or advise medications.
3. Output MUST be ONLY valid JSON with no markdown formatting or commentary.
4. Output must contain EXACTLY these five fields:
   - "sentence_ne": one short, polite Nepali sentence summarizing the signs for the doctor.
   - "complaint": confirmed complaint sign name in English (e.g. "headache") or null if missing.
   - "duration_days": integer duration (2 or 5) or null if missing.
   - "allergy": boolean true/false or null if missing.
   - "medicine_taken": boolean true/false or null if missing.
"""


def call_model(prompt: str) -> str:
    """Call Gemini/Gemma API with temperature=0.0 and automatic transient retry.

    Args:
        prompt: Prompt string sent to the model.

    Returns:
        Generated text string.

    Raises:
        RuntimeError: If API key is missing or model call fails.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not configured.")

    model_name = os.getenv("GEMMA_MODEL", "gemini-2.5-flash")

    # Import google-genai using official client syntax
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(temperature=0.0)

    # Retry once on transient API connection error
    last_err = None
    for attempt in range(2):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config,
            )
            return response.text or ""
        except Exception as err:
            last_err = err
            if attempt == 0:
                time.sleep(1.0)
                continue

    raise RuntimeError(f"Language model API request failed: {type(last_err).__name__}")


def parse_and_validate(text: str, answers: Dict[str, str]) -> Dict[str, Any]:
    """Parse JSON output and strictly validate fields against confirmed answers.

    Args:
        text: Raw text string from model.
        answers: Confirmed answers dictionary.

    Returns:
        Validated dictionary with expected fields.

    Raises:
        ValidationError: If JSON is invalid, fields are missing, or values contradict answers.
    """
    clean_text = text.strip()
    # Strip markdown code fences if present
    if clean_text.startswith("```"):
        clean_text = re.sub(r"^```(?:json)?\s*", "", clean_text)
        clean_text = re.sub(r"\s*```$", "", clean_text)
        clean_text = clean_text.strip()

    try:
        data = json.loads(clean_text)
    except json.JSONDecodeError as err:
        raise ValidationError(f"Invalid JSON response: {err}") from err

    if not isinstance(data, dict):
        raise ValidationError("Model output must be a JSON dictionary.")

    # 1. Key validation: no extra and no missing keys
    extra_keys = set(data.keys()) - ALLOWED_JSON_KEYS
    if extra_keys:
        raise ValidationError(f"Unexpected extra fields in JSON: {extra_keys}")

    missing_keys = ALLOWED_JSON_KEYS - set(data.keys())
    if missing_keys:
        raise ValidationError(f"Missing required fields in JSON: {missing_keys}")

    # 2. Validate sentence_ne
    sentence = data.get("sentence_ne")
    if not isinstance(sentence, str) or not sentence.strip():
        raise ValidationError("Field 'sentence_ne' must be a non-empty string.")

    # 3. Validate complaint
    expected_complaint = answers.get("complaint")
    if data.get("complaint") != expected_complaint:
        raise ValidationError(
            f"Field 'complaint' ({data.get('complaint')}) does not match confirmed sign ({expected_complaint})."
        )

    # 4. Validate duration_days
    exp_days = 2 if answers.get("duration_number") == "two" else (5 if answers.get("duration_number") == "five" else None)
    if data.get("duration_days") != exp_days:
        raise ValidationError(
            f"Field 'duration_days' ({data.get('duration_days')}) does not match confirmed ({exp_days})."
        )

    # 5. Validate allergy
    exp_allergy = True if answers.get("allergy") == "yes" else (False if answers.get("allergy") == "no" else None)
    if data.get("allergy") != exp_allergy:
        raise ValidationError(
            f"Field 'allergy' ({data.get('allergy')}) does not match confirmed ({exp_allergy})."
        )

    # 6. Validate medicine_taken
    exp_med = True if answers.get("medicine_taken") == "yes" else (False if answers.get("medicine_taken") == "no" else None)
    if data.get("medicine_taken") != exp_med:
        raise ValidationError(
            f"Field 'medicine_taken' ({data.get('medicine_taken')}) does not match confirmed ({exp_med})."
        )

    return data


def fallback_sentence(answers: Dict[str, str]) -> Dict[str, Any]:
    """Generate deterministic summary and Nepali sentence from fixed templates.

    Used when model calls fail or produce invalid JSON.

    Args:
        answers: Dictionary of confirmed sign answers.

    Returns:
        Dictionary adhering to the standard structured intake summary format.
    """
    complaint_sign = answers.get("complaint")
    complaint_phrases = {
        "headache": "टाउको दुखाइ छ",
        "stomachache": "पेट दुखाइ छ",
        "fever": "ज्वरो आएको छ",
        "cough": "खोकी लागेको छ",
        "vomiting": "बान्ता भएको छ",
    }

    # Duration clause
    dur_num = answers.get("duration_number")
    dur_clause = ""
    if dur_num == "two":
        dur_clause = "२ दिनदेखि "
    elif dur_num == "five":
        dur_clause = "५ दिनदेखि "

    # Complaint clause
    complaint_clause = complaint_phrases.get(complaint_sign, "समस्या छ") if complaint_sign else ""

    # Secondary clauses
    clauses: list[str] = []
    if answers.get("allergy") == "yes":
        clauses.append("एलर्जी छ")
    elif answers.get("allergy") == "no":
        clauses.append("एलर्जी छैन")

    if answers.get("medicine_taken") == "yes":
        clauses.append("औषधि खाएको छ")
    elif answers.get("medicine_taken") == "no":
        clauses.append("औषधि खाएको छैन")

    if complaint_sign:
        main_part = f"बिरामीलाई {dur_clause}{complaint_clause}"
        if clauses:
            sentence = f"{main_part}, {', '.join(clauses)}।"
        else:
            sentence = f"{main_part}।"
    elif clauses:
        sentence = f"बिरामीको {', '.join(clauses)}।"
    else:
        sentence = "बिरामीको लक्षण विवरण उपलब्ध छैन।"

    return {
        "sentence_ne": sentence,
        "complaint": complaint_sign,
        "duration_days": 2 if dur_num == "two" else (5 if dur_num == "five" else None),
        "allergy": True if answers.get("allergy") == "yes" else (False if answers.get("allergy") == "no" else None),
        "medicine_taken": True if answers.get("medicine_taken") == "yes" else (False if answers.get("medicine_taken") == "no" else None),
    }


def summarise(answers: Dict[str, str]) -> Dict[str, Any]:
    """Generate final clinical summary: tries model with validation retry, falls back gracefully.

    Args:
        answers: Dictionary of confirmed sign answers.

    Returns:
        Summary dict containing sentence_ne, structured fields, used_model flag, and reason.
    """
    prompt = build_prompt(answers)
    last_error: Optional[str] = None

    # Try model call up to 2 times (initial attempt + one validation retry)
    for attempt in range(2):
        try:
            raw_text = call_model(prompt)
            validated = parse_and_validate(raw_text, answers)
            validated["used_model"] = True
            validated["reason"] = None
            return validated
        except Exception as err:
            last_error = f"{type(err).__name__}: {str(err)}"

    # If model fails or raises validation error twice, use deterministic fallback
    result = fallback_sentence(answers)
    result["used_model"] = False
    result["reason"] = last_error
    return result
