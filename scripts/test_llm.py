"""Test script for question flow and Gemma/Gemini language layer.

Tests validation rules offline on synthetic and corrupted model responses,
and optionally runs live API calls against all clinical test cases.

Run from project root:
    python -m scripts.test_llm           # Offline checks (fast, no API key needed)
    python -m scripts.test_llm --live    # Live model calls across 21 clinical test cases
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

from nsl.config import PROJECT_ROOT
from nsl.flow import QUESTIONS, is_valid_answer, next_question
from nsl.llm import ValidationError, fallback_sentence, parse_and_validate, summarise

DEFAULT_CASES_PATH = PROJECT_ROOT / "tests" / "llm_cases.json"


def test_question_flow() -> None:
    """Verify flow sequencing and answer validation logic."""
    assert len(QUESTIONS) == 5
    answers: dict[str, str] = {}

    # Initial question must be complaint
    q1 = next_question(answers)
    assert q1 is not None and q1["id"] == "complaint"
    assert is_valid_answer("complaint", "fever")
    assert not is_valid_answer("complaint", "unknown_sign")

    # Step through all questions
    answers["complaint"] = "fever"
    q2 = next_question(answers)
    assert q2 is not None and q2["id"] == "duration_number"

    answers["duration_number"] = "two"
    answers["duration_unit"] = "day"
    answers["allergy"] = "no"
    answers["medicine_taken"] = "no"

    # All answered: next_question must return None
    assert next_question(answers) is None


def test_validation_rules() -> None:
    """Verify parse_and_validate accepts valid outputs and catches hallucinations."""
    answers = {
        "complaint": "fever",
        "duration_number": "two",
        "duration_unit": "day",
        "allergy": "no",
        "medicine_taken": "no",
    }

    # 1. Valid JSON format
    valid_json = json.dumps({
        "sentence_ne": "बिरामीलाई २ दिनदेखि ज्वरो आएको छ, एलर्जी छैन र औषधि खाएको छैन।",
        "complaint": "fever",
        "duration_days": 2,
        "allergy": False,
        "medicine_taken": False,
    })
    res = parse_and_validate(valid_json, answers)
    assert res["complaint"] == "fever"
    assert res["duration_days"] == 2

    # 2. Markdown fence stripping
    fenced_json = f"```json\n{valid_json}\n```"
    res_fenced = parse_and_validate(fenced_json, answers)
    assert res_fenced["complaint"] == "fever"

    # 3. Deliberately wrong cases must all raise ValidationError
    wrong_cases = [
        # Extra hallucinated symptom / extra key
        json.dumps({
            "sentence_ne": "ज्वरो छ", "complaint": "fever", "duration_days": 2,
            "allergy": False, "medicine_taken": False, "hallucinated_field": "vomiting"
        }),
        # Wrong complaint sign
        json.dumps({
            "sentence_ne": "खोकी छ", "complaint": "cough", "duration_days": 2,
            "allergy": False, "medicine_taken": False
        }),
        # Wrong duration number
        json.dumps({
            "sentence_ne": "ज्वरो छ", "complaint": "fever", "duration_days": 5,
            "allergy": False, "medicine_taken": False
        }),
        # Contradicting allergy flag
        json.dumps({
            "sentence_ne": "ज्वरो छ", "complaint": "fever", "duration_days": 2,
            "allergy": True, "medicine_taken": False
        }),
        # Empty sentence
        json.dumps({
            "sentence_ne": "", "complaint": "fever", "duration_days": 2,
            "allergy": False, "medicine_taken": False
        }),
        # Invalid JSON syntax
        "Invalid non-JSON response string from model",
    ]

    for bad in wrong_cases:
        try:
            parse_and_validate(bad, answers)
            assert False, f"Expected ValidationError on invalid response: {bad}"
        except ValidationError:
            pass


def run_offline_tests(cases_path: Path) -> bool:
    """Run full offline test suite on synthetic cases and templates."""
    print("\n=== RUNNING OFFLINE LANGUAGE-LAYER TESTS ===")

    test_question_flow()
    print("[PASS] Question flow sequencing and answer validation")

    test_validation_rules()
    print("[PASS] Validation rules (extra fields, contradictions, bad JSON)")

    # Test fallback_sentence on every case in llm_cases.json
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    for case in cases:
        out = fallback_sentence(case["answers"])
        assert out["sentence_ne"], f"Empty sentence for case {case['name']}"
        assert out["complaint"] == case["expected"]["complaint"]
        assert out["duration_days"] == case["expected"]["duration_days"]
        assert out["allergy"] == case["expected"]["allergy"]
        assert out["medicine_taken"] == case["expected"]["medicine_taken"]

    print(f"[PASS] Fallback templates verified across all {len(cases)} clinical cases")
    print("\nAll offline checks passed successfully (100%).")
    return True


def run_live_tests(cases_path: Path) -> bool:
    """Execute live model generation and validation on each clinical case."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("\nERROR: GEMINI_API_KEY is not set in environment. Cannot run --live tests.")
        return False

    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    print(f"\n=== RUNNING LIVE MODEL TESTS ({len(cases)} cases) ===")

    passed = 0
    for case in cases:
        answers = case["answers"]
        expected = case["expected"]
        res = summarise(answers)

        # Verify structured outputs match expected clinical facts
        ok_complaint = (res.get("complaint") == expected["complaint"])
        ok_duration = (res.get("duration_days") == expected["duration_days"])
        ok_allergy = (res.get("allergy") == expected["allergy"])
        ok_med = (res.get("medicine_taken") == expected["medicine_taken"])
        ok_sentence = bool(res.get("sentence_ne"))

        is_ok = ok_complaint and ok_duration and ok_allergy and ok_med and ok_sentence
        status = "PASS" if is_ok else "FAIL"
        used = "model" if res.get("used_model") else f"fallback ({res.get('reason')})"
        print(f"[{status}] {case['name']:<42} via {used}")

        if is_ok:
            passed += 1
        else:
            print(f"       Expected: {expected}")
            print(f"       Got:      {res}")

    pct = (100.0 * passed / len(cases)) if cases else 0.0
    print(f"\nFinal Live Score: {passed}/{len(cases)} passed ({pct:.1f}%)")
    return passed == len(cases)


def main() -> None:
    """Parse command line arguments and run offline or live tests."""
    parser = argparse.ArgumentParser(description="Test language layer and clinical question flow.")
    parser.add_argument("--live", action="store_true", help="Call real LLM model for all cases.")
    parser.add_argument(
        "--cases",
        type=Path,
        default=DEFAULT_CASES_PATH,
        help="Path to llm_cases.json test suite.",
    )
    args = parser.parse_args()

    # Always execute offline assertions
    offline_ok = run_offline_tests(args.cases)
    if not offline_ok:
        return

    # Optionally execute live model requests
    if args.live:
        run_live_tests(args.cases)


if __name__ == "__main__":
    main()

