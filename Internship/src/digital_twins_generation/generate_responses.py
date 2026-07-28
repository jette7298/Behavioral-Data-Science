"""Core sandbox call for generating structured digital-twin responses.

The caller must supply an authenticated Azure/OpenAI-compatible client.
Authentication and internal infrastructure are intentionally not included.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def build_persona_payload(row: pd.Series) -> dict[str, Any]:
    """Select only information needed to simulate one anonymized respondent."""
    return {
        "persona_id": str(row["persona_id"]),
        "persona_summary": str(row["persona_summary"]),
    }


def build_prompt(persona: dict[str, Any], questionnaire: list[dict]) -> str:
    """Combine persona, survey, instructions, and expected output structure."""
    output_structure = {
        "persona_id": "same identifier as the input persona",
        "answers": {
            "QUESTION_ID": "option id, number, free-text answer, or null"
        },
    }
    return "\n\n".join(
        [
            "Simulate one respondent from the persona. Return valid JSON only.",
            "Persona:\n" + json.dumps(persona, ensure_ascii=False),
            "Questionnaire:\n" + json.dumps(questionnaire, ensure_ascii=False),
            (
                "Answer every applicable question from the persona's perspective. "
                "Use the supplied option identifiers exactly."
            ),
            "Output structure:\n" + json.dumps(output_structure, ensure_ascii=False),
        ]
    )


def generate_response(
    client,
    deployment_name: str,
    persona: dict[str, Any],
    questionnaire: list[dict],
) -> tuple[dict[str, Any], str]:
    """Make one structured completion call through a preconfigured sandbox client."""
    prompt = build_prompt(persona, questionnaire)
    response = client.chat.completions.create(
        model=deployment_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content
    return json.loads(raw), raw


def flatten_response(
    parsed: dict[str, Any],
    questionnaire: list[dict],
) -> dict[str, Any]:
    """Convert nested answers to one tabular respondent row."""
    answers = parsed.get("answers", {})
    row = {"persona_id": parsed.get("persona_id")}
    for question in questionnaire:
        question_id = question["question_id"]
        value = answers.get(question_id)
        row[question_id] = (
            json.dumps(value, ensure_ascii=False)
            if isinstance(value, (list, dict))
            else value
        )
    return row


def save_response(
    parsed: dict[str, Any],
    raw: str,
    questionnaire: list[dict],
    raw_path: str | Path,
    table_path: str | Path,
) -> None:
    """Store the unmodified model JSON and the flattened analytical row."""
    Path(raw_path).write_text(raw, encoding="utf-8")
    pd.DataFrame([flatten_response(parsed, questionnaire)]).to_csv(
        table_path,
        index=False,
        encoding="utf-8-sig",
    )

