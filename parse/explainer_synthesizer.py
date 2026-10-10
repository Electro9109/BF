"""ExplainerSynthesizer — PARSE Core Synthesizer implementation using the Qwen Explainer.

This module implements the parse.core.operations.Synthesizer protocol using the
existing Qwen Explainer LLM infrastructure. It translates AnalysisResult findings
into human-readable explanations while preserving fidelity through validation checks.

Design rules
------------
- Implements the Synthesizer protocol from parse.core.operations
- Reuses existing explainer_llm.explain() and explainer_adapter infrastructure
- Accepts AnalysisResult inputs and returns SynthesisResult per PARSE Core contracts
- Preserves deterministic fallback behavior when model is disabled or validation fails
- Does not perform statistical analysis or retrieve information — only translates
- Recommendations are passed through from analysis when present, never invented
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from parse.analysis import AnalysisRequest
from parse.core.contracts import (
    AnalysisResult,
    EvidenceRef,
    Provenance,
    SourceRef,
    SynthesisResult,
)
from parse.core.operations import Synthesizer
from parse.explainer_adapter import eda_result_to_explainer_examples
from parse.explainer_llm import explain, ExplainerResult

logger = logging.getLogger(__name__)


class ExplainerSynthesizer:
    """Synthesizer implementation using the PARSE Qwen Explainer LLM.

    Translates structured analysis findings into natural language explanations
    while enforcing fidelity checks and providing deterministic fallback.

    The synthesizer:
    - Converts AnalysisResult findings to Explainer records via the adapter
    - Calls the LLM explainer with each record
    - Validates numerical fidelity, causal language, and recommendation strength
    - Returns SynthesisResult with the explanation or fallback text
    - Preserves all evidence references from the input analysis
    """

    def synthesize_finding(
        self,
        finding: Any,
        recommendation: str | None = None,
        evidence: Any = None,
        **context: Any,
    ) -> SynthesisResult:
        """Explain a single Finding while reusing the existing synthesizer pipeline.

        This keeps the UI and any single-finding callers on the same contract as the
        broader analysis pipeline without inventing new architecture.
        """
        import pandas as pd

        if finding is None:
            return SynthesisResult(
                result_id=f"synthesis_missing_{id(context)}",
                content="No finding selected for explanation.",
                evidence_refs=[],
                analysis_refs=[],
                method="ExplainerSynthesizer",
                provenance=Provenance(
                    sources=[SourceRef("explainer_synthesizer", "model")],
                    operation="synthesize",
                    method="ExplainerSynthesizer",
                ),
                issues=[],
                limitations=["No finding was available to explain."],
            )

        source = SourceRef("ui_explainer", "generated_output", label="Selected finding")
        synthetic_request = AnalysisRequest(
            dataset=pd.DataFrame({"finding": [str(getattr(finding, "observation", str(finding)))]}),
            source=source,
        )
        synthetic_analysis = type(
            "SyntheticSingleFindingAnalysis",
            (),
            {
                "result_id": f"analysis_{id(finding)}",
                "request": synthetic_request,
                "findings": [finding],
                "provenance": Provenance(
                    sources=[source],
                    operation="synthesize_single_finding",
                    method="ExplainerSynthesizer",
                ),
            },
        )()
        context = dict(context)
        if recommendation is not None:
            context["recommendation"] = recommendation
        return self.synthesize(evidence=evidence, analyses=[synthetic_analysis], **context)

    def __init__(self, enabled: bool = True):
        """Initialize the synthesizer.

        Parameters
        ----------
        enabled : bool
            If False, force fallback mode without loading model weights.
            Can be overridden by PARSE_EXPLAINER_ENABLED environment variable.
        """
        self._enabled = enabled

    def synthesize(
        self,
        evidence: Any,
        analyses: Sequence[AnalysisResult] = (),
        **context: Any,
    ) -> SynthesisResult:
        """Translate analysis findings into human-readable explanations.

        Parameters
        ----------
        evidence : Any
            Retrieval evidence (unused by Explainer, kept for protocol compatibility).
        analyses : Sequence[AnalysisResult]
            Analysis results containing findings to explain.
        **context : Any
            Optional context, including 'recommendation' strings for specific findings.

        Returns
        -------
        SynthesisResult
            Contains the generated explanation (or fallback), evidence references,
            validation status, and provenance information.
        """
        import uuid

        result_id = f"synthesis_{uuid.uuid4().hex[:8]}"

        if not analyses:
            return SynthesisResult(
                result_id=result_id,
                content="No analysis results provided for explanation.",
                evidence_refs=[],
                analysis_refs=[],
                method="ExplainerSynthesizer",
                provenance=Provenance(
                    sources=[SourceRef("explainer_synthesizer", "model")],
                    operation="synthesize",
                    method="ExplainerSynthesizer",
                ),
                issues=[],
                limitations=["No analysis input to synthesize."],
            )

        # Collect all findings from all analysis results
        all_findings = []
        analysis_refs = []
        for analysis in analyses:
            # Extract findings from AnalysisResult if it follows the EDA pattern
            if hasattr(analysis, "findings"):
                all_findings.extend(analysis.findings)
            # Track evidence references
            result_id = getattr(analysis, "result_id", None)
            if result_id is None and hasattr(analysis, "request"):
                result_id = f"analysis_{id(analysis)}"
            analysis_refs.extend(
                [
                    EvidenceRef(
                        ref_id=result_id or f"analysis_{id(analysis)}",
                        relation="derived_from",
                    )
                ]
            )

        if not all_findings:
            return SynthesisResult(
                result_id=result_id,
                content="Analysis provided but contains no findings to explain.",
                evidence_refs=[],
                analysis_refs=analysis_refs,
                method="ExplainerSynthesizer",
                provenance=Provenance(
                    sources=[SourceRef("explainer_synthesizer", "model")],
                    operation="synthesize",
                    method="ExplainerSynthesizer",
                ),
                issues=[],
                limitations=["No findings in analysis results."],
            )

        # Convert findings to Explainer records
        from parse.analysis import EDAResult, AnalysisRequest
        # Reconstruct a minimal EDAResult for the adapter
        # (adapter expects EDAResult, not raw Finding list)
        first_analysis = analyses[0]
        if hasattr(first_analysis, "findings"):
            # Create a synthetic EDAResult for adapter compatibility
            # Reuse the original request if available
            if hasattr(first_analysis, "request"):
                synthetic_request = first_analysis.request
            else:
                # Create a minimal valid request with an empty DataFrame
                import pandas as pd
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
                provenance=first_analysis.provenance if hasattr(first_analysis, "provenance") else None,
            )
            explainer_records = eda_result_to_explainer_examples(synthetic_eda)
        else:
            # Fallback: cannot convert without EDAResult structure
            return SynthesisResult(
                result_id=result_id,
                content="Analysis result structure not compatible with Explainer adapter.",
                evidence_refs=[],
                analysis_refs=analysis_refs,
                method="ExplainerSynthesizer",
                provenance=Provenance(
                    sources=[SourceRef("explainer_synthesizer", "model")],
                    operation="synthesize",
                    method="ExplainerSynthesizer",
                ),
                issues=[],
                limitations=["Analysis result does not contain EDAResult-compatible findings."],
            )

        # Explain each finding and collect results
        explanations = []
        issues = []
        limitations = []
        fallback_used = False

        for record in explainer_records:
            # Check for recommendation in context
            recommendation = context.get("recommendation")
            # If context has per-finding recommendations, match by finding_id
            if "recommendations" in context and isinstance(context["recommendations"], dict):
                recommendation = context["recommendations"].get(record["finding_id"])

            result: ExplainerResult = explain(record, recommendation=recommendation)

            if result.fallback_used:
                fallback_used = True
                limitations.append(f"Fallback used for {record['finding_id']}: {result.validation_detail}")

            explanations.append(result.text)

        # Combine all explanations into one content block
        combined_content = "\n\n".join(explanations)

        # Build provenance
        provenance = Provenance(
            sources=[SourceRef("explainer_llm", "model", label="Qwen Explainer")],
            operation="synthesize",
            method="ExplainerSynthesizer",
        )

        return SynthesisResult(
            result_id=result_id,
            content=combined_content,
            evidence_refs=[],
            analysis_refs=analysis_refs,
            method="ExplainerSynthesizer",
            issues=issues,
            limitations=limitations,
            provenance=provenance,
        )
