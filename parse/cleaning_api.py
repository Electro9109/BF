"""Public API for Analysis -> Cleaning integration."""

from __future__ import annotations

import pandas as pd

from parse.cleaning import (
    CleaningIssue,
    CleaningResult,
    DataCleaner,
    HumanDecision,
    TransformationProposal,
    ValidationResult,
)
from parse.cleaning_context import (
    CleaningContext,
    CleaningPurpose,
    create_cleaning_context,
)
from parse.semantic_analysis import AnalysisBundle, HumanContext


def propose_cleaning(
    context: CleaningContext,
    frame: pd.DataFrame,
) -> tuple[list[CleaningIssue], list[TransformationProposal]]:
    """Propose data cleaning operations based on analysis evidence."""
    cleaner = DataCleaner(context.source_ref)
    return cleaner.detect(frame, context=context)


def apply_cleaning(
    frame: pd.DataFrame,
    decisions: list[HumanDecision],
    context: CleaningContext | None = None,
) -> CleaningResult:
    """Apply approved cleaning decisions and perform validation."""
    cleaner = DataCleaner(context.source_ref if context else None)
    return cleaner.clean(frame, decisions=decisions, context=context)


def validate_cleaning(
    original: pd.DataFrame,
    cleaned: pd.DataFrame,
    context: CleaningContext | None = None,
) -> ValidationResult:
    """Perform before/after validation of cleaning effects."""
    cleaner = DataCleaner(context.source_ref if context else None)
    
    before_snap = cleaner._snapshot(original)
    after_snap = cleaner._snapshot(cleaned)
    
    missing_before = sum(before_snap.missing_counts.values())
    missing_after = sum(after_snap.missing_counts.values())
    
    return ValidationResult(
        intended_issues_addressed=[],
        newly_introduced_issues=[],
        before_snapshot=before_snap,
        after_snapshot=after_snap,
        comparison={
            "row_count_diff": after_snap.row_count - before_snap.row_count,
            "duplicate_count_diff": after_snap.duplicate_rows - before_snap.duplicate_rows,
            "total_missing_diff": missing_after - missing_before,
        }
    )

