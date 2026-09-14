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
    
    # We need the original issues to diff against. Since this API takes original/cleaned 
    # directly without prior state, we have to detect original issues now.
    issues, proposals = cleaner.detect(original, context=context)
    
    return cleaner._validate_effects(original, cleaned, issues, proposals)
