"""Static text and icons used by the demo screen.

Edit this file to change wording. The Nepali sentences below were written by an
AI assistant: ask a Nepali speaker to check them before the presentation.
"""

# English labels for each sign (shown under the Nepali word).
SIGN_EN = {
    "headache": "Headache",
    "stomachache": "Stomach ache",
    "fever": "Fever",
    "cough": "Cough",
    "vomiting": "Vomiting",
    "yes": "Yes",
    "no": "No",
    "day": "Day(s)",
    "two": "Two",
    "five": "Five",
    "rest": "(no sign)",
}

# Emoji used as simple pictograms, so the screen does not rely on reading alone.
SIGN_ICON = {
    "headache": "🤕",
    "stomachache": "😖",
    "fever": "🤒",
    "cough": "😷",
    "vomiting": "🤮",
    "yes": "✅",
    "no": "❌",
    "day": "📅",
    "two": "2️⃣",
    "five": "5️⃣",
}

# English version of each question id from nsl/flow.py.
QUESTION_EN = {
    "complaint": "What is the problem?",
    "duration_number": "For how many days?",
    "duration_unit": "Please sign: days",
    "allergy": "Do you have any allergy?",
    "medicine_taken": "Did you already take medicine?",
}

# Fixed doctor replies shown to the patient as large Nepali text.
DOCTOR_REPLIES = [
    {"ne": "यो औषधि दिनमा दुई पटक खानुहोस्।", "en": "Take this medicine twice a day."},
    {"ne": "खाना खाएपछि औषधि खानुहोस्।", "en": "Take the medicine after eating."},
    {"ne": "धेरै पानी पिउनुहोस् र आराम गर्नुहोस्।", "en": "Drink plenty of water and rest."},
    {"ne": "तीन दिनपछि फेरि आउनुहोस्।", "en": "Come back after three days."},
    {"ne": "रगत जाँच गर्नुपर्छ।", "en": "A blood test is needed."},
    {"ne": "कृपया पर्खनुहोस्, म जाँच गर्छु।", "en": "Please wait, I will examine you."},
]


def sign_label(sign: str, sign_ne: dict) -> dict:
    """Return the label dict (sign, Nepali, English, icon) for one sign."""
    return {
        "sign": sign,
        "ne": sign_ne.get(sign, ""),
        "en": SIGN_EN.get(sign, sign.replace("_", " ").title()),
        "icon": SIGN_ICON.get(sign, "🖐️"),
    }
