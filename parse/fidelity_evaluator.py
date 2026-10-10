"""FidelityEvaluator — PARSE Core Evaluator implementation using fidelity checks.

This module implements the parse.core.operations.Evaluator protocol using the
existing fidelity_checks infrastructure. It validates that explanations preserve
numerical values, causal language, and limitations from the source findings.

Design rules
------------
- Implements the Evaluator protocol from parse.core.operations
- Reuses existing eval.fidelity_checks infrastructure
- Accepts SynthesisResult inputs and returns EvaluationResult per PARSE Core contracts
- Does not replace the Analysis Engine — only validates synthesis fidelity
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from parse.analysis import AnalysisRequest, EDAResult
from parse.core.contracts import (
    AnalysisResult,
    EvaluationResult,
    EvidenceRef,
    Provenance,
    SourceRef,
    SynthesisResult,
)
from parse.core.operations import Evaluator

logger = logging.getLogger(__name__)


class FidelityEvaluator:
    """Evaluator implementation using PARSE fidelity checks.

    Validates that synthesized explanations preserve the fidelity of source
    analysis findings, including numerical values, causal language, and
    limitations.

    The evaluator:
    - Extracts findings from analysis results
    - Runs fidelity checks on the synthesized content
    - Returns EvaluationResult with pass/fail status and details
    """

    def evaluate(
        self,
        target: Any,
        criteria: Sequence[str] = (),
        **context: Any,
    ) -> EvaluationResult:
        """Evaluate a synthesis result against fidelity criteria.

        Parameters
        ----------
        target : Any
            The target to evaluate (expected to be SynthesisResult).
        criteria : Sequence[str]
            Specific fidelity criteria to check. If empty, runs all hard checks.
        **context : Any
            Optional context, including analysis results with source findings.

        Returns
        -------
        EvaluationResult
            Contains evaluation metrics, interpretation, and evidence references.
        """
        if not isinstance(target, SynthesisResult):
            return EvaluationResult(
                target_ref=EvidenceRef(ref_id=str(id(target)), relation="derived_from"),
                criteria=list(criteria) or ["numerical_fidelity", "causal_language", "limitation_preserved"],
                metrics={"passed": False, "evaluated": False},
                interpretation="Target is not a SynthesisResult; cannot evaluate fidelity.",
                evidence_refs=[],
                limitations=["Invalid target type for fidelity evaluation."],
            )

        synthesis = target
        analyses = context.get("analyses", [])

        # Extract findings from analyses for fidelity comparison
        all_findings = []
        for analysis in analyses:
            if hasattr(analysis, "findings"):
                all_findings.extend(analysis.findings)

        if not all_findings:
            return EvaluationResult(
                target_ref=EvidenceRef(ref_id=synthesis.result_id or "synthesis", relation="derived_from"),
                criteria=list(criteria) or ["numerical_fidelity", "causal_language", "limitation_preserved"],
                metrics={"passed": True, "evaluated": False},
                interpretation="No source findings available for fidelity comparison.",
                evidence_refs=synthesis.evidence_refs,
                limitations=["No source findings to validate against."],
            )

        # Convert findings to explainer records for fidelity checks
        from parse.explainer_adapter import eda_result_to_explainer_examples
        import pandas as pd

        # Create a minimal EDAResult for adapter compatibility
        # Reuse the original request if available
        if analyses and hasattr(analyses[0], "request"):
            synthetic_request = analyses[0].request
        else:
            synthetic_request = AnalysisRequest(
                dataset=pd.DataFrame(),
                source=SourceRef("synthetic", "generated_output"),
            )
        synthetic_eda = EDAResult(
            request=synthetic_request,
            dataset_profile=None,
            structural_profile=None,
            attributes=[],
            findings=all_findings,
            limitations=[],
            provenance=analyses[0].provenance if analyses else None,
        )
        explainer_records = eda_result_to_explainer_examples(synthetic_eda)

        # Run fidelity checks
        from eval.fidelity_checks import run_all_checks

        total_checks = 0
        passed_checks = 0
        failed_checks = []
        check_details = {}

        hard_checks = {"numerical_fidelity", "causal_language", "limitation_preserved", "recommendation_fidelity", "metric_semantics"}
        checks_to_run = set(criteria) if criteria else hard_checks

        for record in explainer_records:
            checks = run_all_checks(record, synthesis.content)
            for check_name, result in checks.items():
                if check_name in checks_to_run:
                    total_checks += 1
                    if result.passed:
                        passed_checks += 1
                    else:
                        failed_checks.append(f"{record['finding_id']}: {check_name}")
                    check_details[f"{record['finding_id']}_{check_name}"] = {
                        "passed": result.passed,
                        "detail": result.detail,
                    }

        all_passed = total_checks > 0 and passed_checks == total_checks

        interpretation = (
            f"All {total_checks} fidelity checks passed."
            if all_passed
            else f"{passed_checks}/{total_checks} checks passed. Failures: {failed_checks}"
        )

        return EvaluationResult(
            target_ref=EvidenceRef(ref_id=synthesis.result_id or "synthesis", relation="derived_from"),
            criteria=list(checks_to_run),
            metrics={
                "total_checks": total_checks,
                "passed_checks": passed_checks,
                "passed": all_passed,
            },
            interpretation=interpretation,
            evidence_refs=synthesis.evidence_refs,
            limitations=[],
        )
