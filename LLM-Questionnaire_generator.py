"""
LLM-assisted questionnaire drafting for CSCL group formation.

This script collects an instructor brief and uses the OpenAI API to propose:
- questionnaire items (text + response format)
- scoring tag for each item (Similarity or Diversity)
- a suggested weight for each item (0-1), plus a short rationale

Output:
- a machine-readable JSON file that can be reviewed/edited by instructors
- the JSON structure is intentionally aligned with the paper's "instrument drafting"
  and downstream "grouping pipeline" steps.

Notes:
- This script produces a *draft* specification. Instructor review and approval are required.
- Keep prompts and outputs under version control to support reproducibility.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import os
import re
import sys
from dotenv import load_dotenv
load_dotenv()
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

# -----------------------------
# Configuration (edit as needed)
# -----------------------------

DEFAULT_MODEL = "gpt-4.1-2025-04-14"
DEFAULT_MAX_ITEMS = 18
DEFAULT_OUTFILE = "questionnaire_spec.json"

Tag = Literal["Similarity", "Diversity"]
ResponseType = Literal["likert", "single_choice", "multi_choice", "ranking", "short_text", "numeric"]

# -----------------------------
# Data model (JSON schema)
# -----------------------------


@dataclass
class QuestionnaireItem:
    """One questionnaire item in the generated specification."""
    id: str
    text: str
    response_type: ResponseType
    options: Optional[List[str]]  # Required for choice/ranking, None otherwise
    tag: Tag
    weight: float
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class QuestionnaireSpec:
    """Full questionnaire specification."""
    metadata: Dict[str, Any]
    instructor_brief: Dict[str, Any]
    items: List[QuestionnaireItem]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": self.metadata,
            "instructor_brief": self.instructor_brief,
            "items": [it.to_dict() for it in self.items],
        }


# -----------------------------
# Helpers
# -----------------------------


def _clean_text(s: str) -> str:
    """Basic cleanup for safe display and JSON writing."""
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _normalize_weight(w: float) -> float:
    """Clamp weight to [0,1] and round for readability."""
    w = max(0.0, min(1.0, w))
    return float(f"{w:.3f}")


def _require_env(var_name: str) -> str:
    val = os.getenv(var_name, "").strip()
    if not val:
        raise RuntimeError(f"Missing required environment variable: {var_name}")
    return val


def _now_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _input_multiline(prompt: str) -> str:
    """Read multiline text from stdin until an empty line."""
    print(prompt)
    print("(Finish with an empty line.)")
    lines: List[str] = []
    while True:
        line = input()
        if not line.strip():
            break
        lines.append(line)
    return "\n".join(lines).strip()


# -----------------------------
# OpenAI client call
# -----------------------------


def draft_questionnaire_with_openai(
    *,
    model: str,
    instructor_brief: Dict[str, Any],
    max_items: int,
) -> Dict[str, Any]:
    """
    Calls the OpenAI API and returns a raw JSON object that should match
    the expected schema.

    We request strict JSON output to reduce post-processing and improve reproducibility.
    """
    # OpenAI Python SDK v1 style. Install: pip install openai
    # Docs may evolve; keep this minimal and easy to adapt.
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError(
            "openai package not found. Install with: pip install openai"
        ) from e

    api_key = _require_env("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)

    system_msg = (
        "You are an assistant helping an instructor draft a questionnaire for forming student groups "
        "in a design studio / CSCL course. "
        "Return ONLY valid JSON matching the requested schema. "
        "Do not include extra commentary or markdown."
    )

    # We keep the prompt explicit and aligned with the paper:
    # - items
    # - response format
    # - similarity/diversity tag
    # - weight + rationale
    user_msg = {
        "task": "Draft a questionnaire specification for group formation.",
        "constraints": {
            "max_items": max_items,
            "weights": {
                "range": [0.0, 1.0],
                "guidance": "Use higher weights for items most important to the instructor's brief. "
                            "Weights do not need to sum to 1; the instructor may renormalize later."
            },
            "tags": ["Similarity", "Diversity"],
            "response_types": ["likert", "single_choice", "multi_choice", "ranking", "short_text", "numeric"],
            "style": "Questions should be concise, unambiguous, and appropriate for students."
        },
        "instructor_brief": instructor_brief,
        "required_json_schema": {
            "items": [
                {
                    "id": "Q1",
                    "text": "string",
                    "response_type": "likert|single_choice|multi_choice|ranking|short_text|numeric",
                    "options": ["string"]  # null/omitted if not applicable
                    ,
                    "tag": "Similarity|Diversity",
                    "weight": 0.0,
                    "rationale": "string"
                }
            ]
        }
    }

    # Request JSON output. We use response_format where available.
    # If your SDK version does not support response_format, remove it and parse manually.
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": json.dumps(user_msg)},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )

    raw = resp.choices[0].message.content
    if not raw:
        raise RuntimeError("Empty response from OpenAI API.")

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError("Model did not return valid JSON.") from e


# -----------------------------
# Validation & conversion
# -----------------------------


def validate_and_convert(raw_json: Dict[str, Any]) -> List[QuestionnaireItem]:
    """
    Validates the model output and converts it to QuestionnaireItem objects.

    This is deliberately strict to keep the repository output predictable.
    """
    if "items" not in raw_json or not isinstance(raw_json["items"], list):
        raise ValueError("Invalid JSON: missing 'items' list.")

    items: List[QuestionnaireItem] = []
    for idx, it in enumerate(raw_json["items"], start=1):
        if not isinstance(it, dict):
            raise ValueError(f"Invalid item at index {idx}: must be an object.")

        # Required fields
        qid = _clean_text(str(it.get("id", f"Q{idx}")))
        text = _clean_text(str(it.get("text", "")))
        response_type = it.get("response_type", "")
        tag = it.get("tag", "")
        weight = it.get("weight", None)
        rationale = _clean_text(str(it.get("rationale", "")))

        if not text:
            raise ValueError(f"Item {qid}: missing text.")
        if response_type not in ("likert", "single_choice", "multi_choice", "ranking", "short_text", "numeric"):
            raise ValueError(f"Item {qid}: invalid response_type '{response_type}'.")
        if tag not in ("Similarity", "Diversity"):
            raise ValueError(f"Item {qid}: invalid tag '{tag}'.")
        if weight is None:
            raise ValueError(f"Item {qid}: missing weight.")
        try:
            weight_f = _normalize_weight(float(weight))
        except Exception as e:
            raise ValueError(f"Item {qid}: invalid weight '{weight}'.") from e
        if not rationale:
            # We allow empty rationale but discourage it
            rationale = "No rationale provided."

        # Options logic
        options = it.get("options", None)

        if response_type in ("single_choice", "multi_choice", "ranking"):
            if not isinstance(options, list):
                options = []
            options_clean = [_clean_text(str(o)) for o in options]
            options_clean = [o for o in options_clean if o]
            if len(options_clean) < 2:
                print(f"Warning: Item {qid} missing valid options; skipping this item.", file=sys.stderr)
                continue
            options = options_clean
        else:
            options = None

        items.append(
            QuestionnaireItem(
                id=qid,
                text=text,
                response_type=response_type,  # type: ignore
                options=options,
                tag=tag,  # type: ignore
                weight=weight_f,
                rationale=rationale,
            )
        )

    return items


def build_spec(
    *,
    instructor_brief: Dict[str, Any],
    items: List[QuestionnaireItem],
    model: str,
) -> QuestionnaireSpec:
    metadata = {
        "generated_at_utc": _now_iso(),
        "generator": "llm_instrument_drafting.py",
        "llm_provider": "OpenAI",
        "llm_model": model,
        "notes": (
            "This file is a draft specification. Instructor review and edits are required "
            "before deploying the questionnaire and running the grouping pipeline."
        ),
    }
    return QuestionnaireSpec(metadata=metadata, instructor_brief=instructor_brief, items=items)


# -----------------------------
# CLI: collect instructor brief
# -----------------------------


def collect_instructor_brief() -> Dict[str, Any]:
    """
    Collects a structured brief from the instructor.

    Keep this short. The goal is to capture intent, constraints, and priorities.
    """
    print("\n--- Instructor brief ---")
    course_title = input("Course/studio name (optional): ").strip()
    cohort_size = input("Cohort size (approx.): ").strip()
    duration_weeks = input("Course duration in weeks (approx., optional): ").strip()

    goals = _input_multiline(
        "\nDescribe the course goals and deliverables (short paragraph)."
    )
    collaboration_priorities = _input_multiline(
        "\nDescribe what you want from collaboration (e.g., fairness, speed, creativity, interdisciplinarity)."
    )
    constraints = _input_multiline(
        "\nList constraints for grouping (e.g., skill coverage, language mix, distribute leaders, avoid cliques)."
    )
    avoid = _input_multiline(
        "\nList outcomes to avoid (e.g., single-skill groups, friendship cliques, isolated international students)."
    )

    # Optional: define key skills or dimensions to distribute
    key_dimensions = input(
        "\nKey diversity dimensions to consider (comma-separated, optional): "
    ).strip()

    brief = {
        "course_title": course_title or None,
        "cohort_size": cohort_size or None,
        "duration_weeks": duration_weeks or None,
        "course_goals_and_deliverables": goals,
        "collaboration_priorities": collaboration_priorities,
        "grouping_constraints": constraints,
        "outcomes_to_avoid": avoid,
        "key_diversity_dimensions": [d.strip() for d in key_dimensions.split(",") if d.strip()] or None,
    }
    return brief


def main() -> int:
    parser = argparse.ArgumentParser(
        description="LLM-assisted questionnaire drafting for group formation."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model name/ID.")
    parser.add_argument("--max-items", type=int, default=DEFAULT_MAX_ITEMS, help="Maximum number of questionnaire items.")
    parser.add_argument("--out", default=DEFAULT_OUTFILE, help="Output JSON file path.")
    args = parser.parse_args()

    brief = collect_instructor_brief()

    print("\nGenerating questionnaire draft via OpenAI API...")
    raw_json = draft_questionnaire_with_openai(
        model=args.model,
        instructor_brief=brief,
        max_items=args.max_items,
    )

    items = validate_and_convert(raw_json)
    spec = build_spec(instructor_brief=brief, items=items, model=args.model)

    out_path = args.out
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(spec.to_dict(), f, indent=2, ensure_ascii=False)

    print(f"\nSaved draft questionnaire specification to: {out_path}")
    print("Next step: open the JSON file, review/edit items and weights, then distribute the questionnaire.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())