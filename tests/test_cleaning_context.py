"""Direct tests for parse/cleaning_context.py -- Task 19.

Correction to the original Task 19 audit: create_cleaning_context() is
already well exercised indirectly via parse.cleaning_api's re-export,
across 7 tests in tests/test_cleaning_integration.py (purpose handling,
missingness mapping, anomaly/outlier enrichment, duplicate-kind
distinction, human decision rejection, before/after validation, downstream
impact). That coverage is real and this file does not duplicate it.

What's genuinely untested, confirmed by reading create_cleaning_context's
branches against every existing test's fixtures: (1) the human_context ->
semantic_status enrichment branch (USER_CONFIRMED / CONFLICTING / INFERRED
/ UNKNOWN), and (2) the relationship-finding enrichment branch (tuple
subject). Both are exercised here with real AnalysisPipeline output, not
hand-built fakes.
"""

import pandas as pd

from parse.analysis import AnalysisRequest
from parse.cleaning_context import create_cleaning_context
from parse.core.contracts import SourceRef
from parse.semantic_analysis import AnalysisPipeline, ConfirmationRecord, HumanContext


def _semantic_bundle():
    frame = pd.DataFrame({
        "record_id": [1, 2, 3],
        "temperature_kg": [10.0, 11.0, 12.0],
        "comment": ["a", "b", "c"],
    })
    request = AnalysisRequest(frame, SourceRef("dataset-semantic", "user_input"))
    return AnalysisPipeline().analyze(request)


def test_semantic_status_defaults_to_inferred_without_human_context():
    bundle = _semantic_bundle()
    context = create_cleaning_context(bundle, human_context=None)

    profile = context.attribute_profiles["temperature_kg"]
    assert profile.semantic_candidates
    assert profile.semantic_status == "INFERRED"


def test_semantic_status_becomes_user_confirmed_after_human_confirmation():
    bundle = _semantic_bundle()
    candidate = next(
        c for c in bundle.semantic.candidates if c.candidate_id == "temperature_kg:unit"
    )

    human_context = HumanContext()
    human_context.apply(ConfirmationRecord(
        "review-1", candidate.candidate_id, "confirm", "temperature_kg", "temperature", "reviewer",
    ))

    context = create_cleaning_context(bundle, human_context=human_context)
    profile = context.attribute_profiles["temperature_kg"]

    assert profile.semantic_status == "USER_CONFIRMED"


def test_semantic_status_becomes_conflicting_after_human_conflict():
    bundle = _semantic_bundle()
    candidate = next(
        c for c in bundle.semantic.candidates if c.candidate_id == "temperature_kg:unit"
    )

    human_context = HumanContext()
    human_context.apply(ConfirmationRecord(
        "review-1", candidate.candidate_id, "conflict", "temperature_kg", None, "reviewer",
    ))

    context = create_cleaning_context(bundle, human_context=human_context)
    profile = context.attribute_profiles["temperature_kg"]

    assert profile.semantic_status == "CONFLICTING"


def test_relationship_findings_are_ported_to_both_attribute_profiles():
    frame = pd.DataFrame({
        "target": [1, 2, 3, 4, 5, 6, 7, 8],
        "support": [2, 4, 6, 8, 10, 12, 14, 16],
        "other": [4, 1, 4, 1, 4, 1, 4, 1],
    })
    request = AnalysisRequest(frame, SourceRef("dataset-relationship", "user_input"))
    bundle = AnalysisPipeline().analyze(request)

    context = create_cleaning_context(bundle)

    assert context.attribute_profiles["target"].relationships
    assert context.attribute_profiles["support"].relationships