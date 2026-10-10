"""Static text and icons used by the demo screen.

Edit this file to change wording. The Nepali sentences below were written by an
AI assistant: ask a Nepali speaker to check them before the presentation.
"""

# Nepali labels for the questions shown in the doctor's confirmed-answer list.
QUESTION_LABEL_NE = {
    "complaint": "मुख्य समस्या",
    "duration_number": "अवधि",
    "allergy": "एलर्जी",
    "medicine_taken": "औषधि सेवन",
}

# Preset doctor replies are written only in Nepali for the patient-facing screen.
DOCTOR_REPLIES = [
    {"ne": "यो औषधि दिनमा दुई पटक खानुहोस्।"},
    {"ne": "खाना खाएपछि औषधि खानुहोस्।"},
    {"ne": "धेरै पानी पिउनुहोस् र आराम गर्नुहोस्।"},
    {"ne": "तीन दिनपछि फेरि आउनुहोस्।"},
    {"ne": "रगत जाँच गर्नुपर्छ।"},
    {"ne": "कृपया पर्खनुहोस्, म जाँच गर्छु।"},
]


def sign_label(sign: str, sign_ne: dict) -> dict:
    """Return the Nepali display label for a sign."""
    return {
        "sign": sign,
        "ne": sign_ne.get(sign, ""),
    }
