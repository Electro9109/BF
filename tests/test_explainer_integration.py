"""End-to-end integration test for Processing -> Analysis -> Explainer -> Evaluation workflow.

This test verifies that:
1. AnalysisOrchestrator produces findings
2. ExplainerSynthesizer translates findings to explanations
3. FidelityEvaluator validates the explanation
4. Fallback behavior works when the model is disabled
"""

import os
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from parse import AnalysisOrchestrator, AnalysisRequest, ExplainerSynthesizer, FidelityEvaluator
from parse.core.contracts import SourceRef, SynthesisResult, EvidenceRef, Provenance
from parse.explainer_adapter import finding_to_explainer_example
from parse.explainer_llm import ExplainerResult


@pytest.fixture
def sample_dataset():
    """Create a sample dataset with planted data quality issues."""
    rng = np.random.default_rng(42)
    n = 50
    df = pd.DataFrame({
        "temperature": rng.normal(1450, 25, n).tolist(),
        "pressure": rng.normal(2.5, 0.3, n).tolist(),
        "batch_id": range(1001, 1001 + n),
    })
    # Plant outliers
    df.loc[3, "temperature"] = 1750.0
    df.loc[7, "pressure"] = 4.5
    # Plant missing values
    df.loc[10:12, "temperature"] = np.nan
    return df


def test_analysis_to_explainer_to_evaluation_workflow(sample_dataset):
    """Test the full workflow from analysis to explanation to evaluation."""
    # Step 1: Analysis
    source = SourceRef("test_integration", "user_input", label="Integration Test")
    request = AnalysisRequest(sample_dataset, source)
    orchestrator = AnalysisOrchestrator()
    analysis_result = orchestrator.analyze(request)

    # Verify analysis produced findings
    assert len(analysis_result.findings) > 0, "Analysis should produce findings"

    # Step 2: Synthesis with Explainer (disabled for deterministic test)
    os.environ["PARSE_EXPLAINER_ENABLED"] = "0"
    synthesizer = ExplainerSynthesizer()
    synthesis = synthesizer.synthesize(
        evidence=None,
        analyses=[analysis_result],
    )

    # Verify synthesis completed (fallback mode)
    assert synthesis.content is not None
    assert synthesis.method == "ExplainerSynthesizer"
    assert len(synthesis.analysis_refs) > 0

    # Step 3: Evaluation
    evaluator = FidelityEvaluator()
    evaluation = evaluator.evaluate(
        target=synthesis,
        criteria=["numerical_fidelity", "causal_language"],
        analyses=[analysis_result],
    )

    # Verify evaluation completed
    assert evaluation.metrics is not None
    assert "passed" in evaluation.metrics
    assert evaluation.interpretation is not None


def test_explainer_synthesizer_with_no_findings(sample_dataset):
    """Test that synthesizer handles empty findings gracefully."""
    source = SourceRef("test_empty", "user_input")
    request = AnalysisRequest(sample_dataset, source)
    orchestrator = AnalysisOrchestrator()
    analysis_result = orchestrator.analyze(request)

    # Clear findings artificially
    from parse.analysis import EDAResult
    empty_analysis = EDAResult(
        request=analysis_result.request,
        dataset_profile=analysis_result.dataset_profile,
        structural_profile=analysis_result.structural_profile,
        attributes=analysis_result.attributes,
        findings=[],
        limitations=analysis_result.limitations,
        provenance=analysis_result.provenance,
    )

    synthesizer = ExplainerSynthesizer()
    synthesis = synthesizer.synthesize(
        evidence=None,
        analyses=[empty_analysis],
    )

    assert "no findings" in synthesis.content.lower()


def test_explainer_synthesizer_with_no_analyses():
    """Test that synthesizer handles empty analysis list gracefully."""
    synthesizer = ExplainerSynthesizer()
    synthesis = synthesizer.synthesize(evidence=None, analyses=[])

    assert "no analysis" in synthesis.content.lower()


def test_fidelity_evaluator_with_non_synthesis_target():
    """Test that evaluator handles non-SynthesisResult gracefully."""
    evaluator = FidelityEvaluator()
    evaluation = evaluator.evaluate(target="not a synthesis")

    assert evaluation.metrics["passed"] is False
    assert "synthesisresult" in evaluation.interpretation.lower()


def test_fidelity_evaluator_with_no_source_findings(sample_dataset):
    """Test that evaluator handles missing source findings gracefully."""
    source = SourceRef("test_no_findings", "user_input")
    request = AnalysisRequest(sample_dataset, source)
    orchestrator = AnalysisOrchestrator()
    analysis_result = orchestrator.analyze(request)

    # Clear findings
    from parse.analysis import EDAResult
    empty_analysis = EDAResult(
        request=analysis_result.request,
        dataset_profile=analysis_result.dataset_profile,
        structural_profile=analysis_result.structural_profile,
        attributes=analysis_result.attributes,
        findings=[],
        limitations=analysis_result.limitations,
        provenance=analysis_result.provenance,
    )

    from parse.core.contracts import SynthesisResult, Provenance
    synthesis = SynthesisResult(
        result_id="test_synthesis",
        content="Test explanation",
        evidence_refs=[],
        analysis_refs=[],
        method="test",
        provenance=Provenance(
            sources=[SourceRef("test", "generated_output")],
            operation="test",
            method="test",
        ),
    )

    evaluator = FidelityEvaluator()
    evaluation = evaluator.evaluate(
        target=synthesis,
        analyses=[empty_analysis],
    )

    assert "no source findings" in evaluation.interpretation.lower()
    assert evaluation.metrics["evaluated"] is False
    assert evaluation.metrics["passed"] is False
    assert synthesis.status == "fallback"
    assert synthesis.validation_passed is False


def test_fidelity_evaluator_checks_separately_supplied_recommendation(sample_dataset):
    source = SourceRef("test_recommendation_fidelity", "user_input")
    analysis_result = AnalysisOrchestrator().analyze(AnalysisRequest(sample_dataset, source))
    finding = analysis_result.findings[0]
    record = finding_to_explainer_example(finding)
    recommendation = f"Consider removing {record['attributes'].get('column', 'batch_id')} from modelling."
    synthesis = SynthesisResult(
        result_id="test_recommendation_synthesis",
        content="The finding is noted.",
        evidence_refs=[],
        analysis_refs=[],
        method="test",
        provenance=Provenance(
            sources=[SourceRef("test", "generated_output")],
            operation="test",
            method="test",
        ),
        finding_id=finding.finding_id,
    )

    evaluation = FidelityEvaluator().evaluate(
        target=synthesis,
        analyses=[analysis_result],
        record=record,
        recommendation=recommendation,
        criteria=["recommendation_fidelity"],
    )

    assert evaluation.metrics["passed"] is False
    assert synthesis.status == "fallback"


def test_fidelity_evaluator_checks_exact_selected_record(sample_dataset):
    source = SourceRef("test_selected_record", "user_input")
    analysis_result = AnalysisOrchestrator().analyze(AnalysisRequest(sample_dataset, source))
    finding = analysis_result.findings[0]
    selected_record = finding_to_explainer_example(finding)
    selected_record["message"] = "The selected finding reports 777 affected records."
    selected_record["attributes"] = {"count": 777}
    synthesis = SynthesisResult(
        result_id="test_selected_record_synthesis",
        content=finding_to_explainer_example(finding)["message"],
        evidence_refs=[],
        analysis_refs=[],
        method="test",
        provenance=Provenance(
            sources=[SourceRef("test", "generated_output")],
            operation="test",
            method="test",
        ),
        finding_id=finding.finding_id,
    )

    evaluation = FidelityEvaluator().evaluate(
        target=synthesis,
        analyses=[analysis_result],
        record=selected_record,
        criteria=["numerical_fidelity"],
    )

    assert evaluation.metrics["passed"] is False
    assert synthesis.status == "fallback"


def test_explainer_synthesizer_single_finding_forwards_recommendation(sample_dataset, monkeypatch):
    """The single-finding explainer path must preserve the recommendation passed separately."""
    monkeypatch.delenv("PARSE_EXPLAINER_ENABLED", raising=False)
    source = SourceRef("test_single_finding", "user_input")
    request = AnalysisRequest(sample_dataset, source)
    orchestrator = AnalysisOrchestrator()
    analysis_result = orchestrator.analyze(request)
    finding = analysis_result.findings[0]

    synthesizer = ExplainerSynthesizer()
    with patch("parse.explainer_synthesizer.explain", return_value=ExplainerResult(text="ok", fallback_used=False)) as mock_explain:
        synthesis = synthesizer.synthesize_finding(finding, recommendation="Consider removing batch_id.")

    assert synthesis.content == "ok"
    assert mock_explain.call_args.kwargs["recommendation"] == "Consider removing batch_id."


def test_explainer_synthesizer_single_finding_uses_fallback_when_disabled(sample_dataset, monkeypatch):
    """Single-finding calls must keep the original finding when the explainer is disabled."""
    monkeypatch.setenv("PARSE_EXPLAINER_ENABLED", "0")
    source = SourceRef("test_single_fallback", "user_input")
    request = AnalysisRequest(sample_dataset, source)
    orchestrator = AnalysisOrchestrator()
    analysis_result = orchestrator.analyze(request)
    finding = analysis_result.findings[0]

    from parse.explainer_adapter import finding_to_explainer_example

    synthesizer = ExplainerSynthesizer()
    synthesis = synthesizer.synthesize_finding(finding, recommendation="Consider removing batch_id.")

    assert synthesis.content == finding_to_explainer_example(finding)["message"]


def test_explainer_synthesizer_environment_kill_switch_takes_precedence(monkeypatch):
    """A PARSE_EXPLAINER_ENABLED=0 kill switch should override an explicit enable flag."""
    monkeypatch.setenv("PARSE_EXPLAINER_ENABLED", "0")
    assert ExplainerSynthesizer(enabled=True).is_enabled() is False
    assert ExplainerSynthesizer(enabled=False).is_enabled() is False


def test_explainer_synthesizer_incompatible_analysis_falls_back_cleanly(sample_dataset):
    """Unsupported or empty analysis inputs must return a controlled fallback instead of crashing."""
    synthesis = ExplainerSynthesizer().synthesize(evidence=None, analyses=[object()])

    assert synthesis.status == "fallback"
    assert synthesis.validation_passed is False
    assert "no findings" in synthesis.content.lower()
    assert synthesis.result_id.startswith("synthesis_")


def test_similarity_cache_clear_function():
    """Test that the similarity cache clear function works."""
    from ml.similarity import clear_training_matrix_cache, _get_cached_training_matrix

    # Load once to populate cache
    import numpy as np
    query = np.array([[0.5] * 15])  # BF feature dimension
    _get_cached_training_matrix()

    # Clear cache
    clear_training_matrix_cache()

    # Verify cache is cleared by checking module-level variable
    from ml import similarity
    assert similarity._cached_training_matrix is None
