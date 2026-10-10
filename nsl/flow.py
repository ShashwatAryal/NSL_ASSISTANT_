"""Question flow and dialogue management for the NSL medical assistant.

Defines the structured clinical intake questionnaire:
1. Primary complaint (headache, stomachache, fever, cough, vomiting)
2. Duration in days (two, five)
3. Known allergies (yes, no)
4. Prior medicine taken (yes, no)
"""

from typing import Dict, List, Optional

# Ordered list of clinical intake questions for the patient
QUESTIONS: List[Dict] = [
    {
        "id": "complaint",
        "text_ne": "तपाईंलाई के समस्या छ?",
        "allowed_signs": ["headache", "stomachache", "fever", "cough", "vomiting"],
    },
    {
        "id": "duration_number",
        "text_ne": "कति दिनदेखि भएको हो?",
        "allowed_signs": ["two", "five"],
    },
    {
        "id": "allergy",
        "text_ne": "तपाईंलाई कुनै एलर्जी छ?",
        "allowed_signs": ["yes", "no"],
    },
    {
        "id": "medicine_taken",
        "text_ne": "तपाईंले कुनै औषधि खानुभएको छ?",
        "allowed_signs": ["yes", "no"],
    },
]


def next_question(answers: Dict[str, str]) -> Optional[Dict]:
    """Return the next unanswered question in the intake sequence.

    Args:
        answers: Dictionary of answers confirmed so far (mapping question id to sign).

    Returns:
        The next question dictionary, or None if all questions have been answered.
    """
    for q in QUESTIONS:
        if q["id"] not in answers:
            return q
    return None


def is_valid_answer(question_id: str, sign: str) -> bool:
    """Check if a recognized sign is an allowed response for a given question.

    Args:
        question_id: Identifier of the question (e.g. 'complaint', 'allergy').
        sign: Recognized sign name (e.g. 'fever', 'yes').

    Returns:
        True if the sign is permitted for this question, False otherwise.
    """
    for q in QUESTIONS:
        if q["id"] == question_id:
            return sign in q["allowed_signs"]
    return False
