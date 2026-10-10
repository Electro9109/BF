"""End-to-end integration test for Processing -> Analysis -> Explainer -> Evaluation workflow.

This test verifies that:
1. AnalysisOrchestrator produces findings
2. ExplainerSynthesizer translates findings to explanations
3. FidelityEvaluator validates the explanation
4. Fallback behavior works when the model is disabled
"""

import os

import numpy as np
import pandas as pd
import pytest

from parse import AnalysisOrchestrator, AnalysisRequest, ExplainerSynthesizer, FidelityEvaluator
from parse.core.contracts import SourceRef


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
