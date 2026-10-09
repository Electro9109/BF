"""Tests for parse/explainer_prompt.py.

Covers:
- Prompt structure and required sections
- Recommendation handling (present vs absent)
- Known failure-case guardrails (recommendation strength, stat meaning)
- Prompt-echo protection: build_user_message must NOT include model output
"""

from __future__ import annotations

import pytest

from parse.explainer_prompt import (
    PROMPT_VERSION,
    _SYSTEM_INSTRUCTION,
    build_chat_messages,
    build_user_message,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def outlier_record():
    return {
        "schema_version": "1.0",
        "finding_id": "outliers_measurement",
        "source": "eda",
        "category": "statistical",
        "kind": "observation",
        "message": "Column 'measurement' contains 3 IQR-based statistical outlier(s).",
        "attributes": {"count": 3, "column": "measurement"},
        "limitations": ["A statistical outlier is not automatically a data error."],
    }


@pytest.fixture
def identifier_record():
    return {
        "schema_version": "1.0",
        "finding_id": "identifier_batch_id",
        "source": "eda",
        "category": "logical",
        "kind": "interpretation",
        "message": "Column 'batch_id' appears to be an identifier or administrative column.",
        "attributes": {"column": "batch_id"},
        "limitations": ["Identifier role is inferred from cardinality; confirm domain meaning."],
    }


@pytest.fixture
def skew_record():
    return {
        "schema_version": "1.0",
        "finding_id": "skew_gas_flow",
        "source": "eda",
        "category": "statistical",
        "kind": "observation",
        "message": "Column 'gas_flow' is strongly skewed (skewness 1.87).",
        "attributes": {"skewness": 1.87, "column": "gas_flow"},
        "limitations": [],
    }


# ── Structure tests ────────────────────────────────────────────────────────────


def test_build_chat_messages_returns_two_turns(outlier_record):
    msgs = build_chat_messages(outlier_record)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"


def test_system_message_contains_strict_rules(outlier_record):
    msgs = build_chat_messages(outlier_record)
    system = msgs[0]["content"]
    # Core constraints must be encoded in the system turn
    assert "do not invent" in system.lower() or "not invent" in system.lower()
    assert "causal" in system.lower()
    assert "recommendation" in system.lower()


def test_user_message_contains_finding_message(outlier_record):
    user = build_user_message(outlier_record)
    assert "Column 'measurement' contains 3 IQR-based statistical outlier(s)." in user


def test_user_message_contains_category_and_kind(outlier_record):
    user = build_user_message(outlier_record)
    assert "statistical" in user
    assert "observation" in user


def test_user_message_contains_attributes(outlier_record):
    user = build_user_message(outlier_record)
    assert "count" in user
    assert "3" in user


def test_user_message_contains_limitations(outlier_record):
    user = build_user_message(outlier_record)
    assert "statistical outlier is not automatically a data error" in user


def test_user_message_with_no_limitations_says_none(skew_record):
    user = build_user_message(skew_record)
    # limitations section must still be present, but say "none"
    assert "limitations" in user.lower()
    assert "none" in user.lower()


# ── Recommendation handling ────────────────────────────────────────────────────


def test_recommendation_included_when_provided(identifier_record):
    rec = "Consider removing batch_id from modelling."
    user = build_user_message(identifier_record, recommendation=rec)
    assert "Consider removing batch_id from modelling." in user
    assert "recommendation" in user.lower()


def test_recommendation_absent_when_not_provided(outlier_record):
    user = build_user_message(outlier_record)
    # When no recommendation is passed, no recommendation section should appear
    assert "recommendation:" not in user.lower()


def test_recommendation_not_invented_for_no_rec_finding(skew_record):
    """If recommendation is not passed, it must not appear in the prompt."""
    user = build_user_message(skew_record, recommendation=None)
    assert "recommendation:" not in user.lower()


# ── Failure case A: Recommendation strength preservation ──────────────────────


def test_recommendation_strength_preserved_verbatim(identifier_record):
    """'Consider removing' must appear exactly in the prompt (not 'remove')."""
    rec = "Consider removing batch_id from modelling."
    user = build_user_message(identifier_record, recommendation=rec)
    assert "Consider removing batch_id from modelling." in user
    # Ensure the weakened form is not substituted
    # (The prompt must carry the exact string so the model sees it)
    assert "Consider removing" in user


# ── Failure case B: Statistical meaning ───────────────────────────────────────


def test_iqr_term_preserved_in_prompt(outlier_record):
    """IQR must be preserved in the message sent to the model, not rewritten as 'range'."""
    user = build_user_message(outlier_record)
    assert "IQR" in user


def test_skewness_value_preserved_in_prompt(skew_record):
    """Skewness value must appear exactly in the prompt."""
    user = build_user_message(skew_record)
    assert "1.87" in user


# ── Failure case C: Prompt-echo protection ────────────────────────────────────


def test_prompt_echo_protection_no_model_output_in_user_turn(outlier_record):
    """The user message must contain ONLY the finding — no pre-generated explanation.

    This ensures that when the eval harness slices off generated tokens,
    it is not accidentally scoring the prompt itself.
    """
    user = build_user_message(outlier_record)
    # The user turn must not contain model-style preamble text
    assert "<|im_start|>assistant" not in user
    assert "<|im_end|>" not in user


def test_chat_messages_no_assistant_turn_prefilled(outlier_record):
    """build_chat_messages must NOT include a pre-filled assistant turn."""
    msgs = build_chat_messages(outlier_record)
    roles = [m["role"] for m in msgs]
    assert "assistant" not in roles


# ── Prompt version ─────────────────────────────────────────────────────────────


def test_prompt_version_is_v03():
    assert PROMPT_VERSION == "v0.3"
