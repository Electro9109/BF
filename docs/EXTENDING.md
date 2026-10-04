# Extending PARSE

This guide outlines how to extend the PARSE system across three key integration points:
1. Adding a new industrial Department
2. Adding a new Analysis method to Core EDA
3. Integrating the Explainer LLM as a `Synthesizer`

---

## 1. Adding a New Department

PARSE separates domain-agnostic analysis and pipeline logic from domain-specific rules. All department-specific logic resides in `departments/<department_name>/`.

### The `Department` Protocol

A new department must implement the `Department` protocol defined in `parse/core/department.py`:

```python
from typing import Protocol, runtime_checkable
import pandas as pd

@runtime_checkable
class Department(Protocol):
    @property
    def department_id(self) -> str:
        """Unique identifier (e.g. 'coke_oven', 'steelmaking')."""
        ...

    @property
    def feature_columns(self) -> list[str]:
        """Raw feature column names (e.g. chemical compositions)."""
        ...

    @property
    def target_columns(self) -> list[str]:
        """Prediction targets (e.g. ['yield', 'temperature'])."""
        ...

    @property
    def value_ranges(self) -> dict[str, tuple[float, float]]:
        """Domain practical ranges for scaling: column -> (min, max)."""
        ...

    def load_data(self, path: str) -> pd.DataFrame:
        """Load and preprocess department-specific raw data."""
        ...

    def parse_custom_fields(self, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Parse department string columns into structured numeric DataFrames."""
        ...
```

### Steps to Implement

1. **Create the department directory:**
   ```
   departments/<department_name>/
   ├── __init__.py
   ├── department.py             # Class implementing Department protocol
   ├── config.py                 # Domain constants, feature lists, ranges
   ├── schemas.py                # Department-specific record dataclasses
   └── feature_processing.py     # Feature engineering & scaling
   ```
2. **Implement `Department` in `department.py`:**
   Reference `departments/blast_furnace/department.py` (`BlastFurnaceDepartment`) as the reference implementation.
3. **Register or Inject into Pipelines:**
   Instantiate the department class and pass it to feature builders or predictor pipelines rather than hardcoding module imports.

---

## 2. Adding an Analysis Method to EDA

PARSE Core data analysis (`parse/analysis.py`) operates as a pipeline of explainable method selectors and profile extractors.

### Where Methods Live

- **Selector Policy (`AnalysisMethodSelector`):**
  Defines when a method is suitable given sample sizes and statistical criteria.
  Example:
  ```python
  def select_numeric_relationship(self, sample_size: int, pearson: float, spearman: float) -> MethodSelection:
      if sample_size < 3:
          return MethodSelection("insufficient_data", "numeric relationship", "Fewer than three complete pairs are available.")
      if abs(pearson) >= .8:
          return MethodSelection("pearson", "numeric relationship", "The observed linear association is strong.",
                                 ("Pairwise complete observations are used.",))
      return MethodSelection("spearman", "numeric relationship", "A rank-based monotonic measure is less dependent on linearity.",
                             ("This remains an association, not a causal test.",))
  ```

- **Analysis Orchestration (`AnalysisOrchestrator`):**
  Orchestrates profiling and constructs traceable `Finding` records:
  ```python
  Finding(
      finding_id=f"method_name:{subject}",
      category="relationship",      # "quality" | "distribution" | "relationship" | "temporal" | "structure"
      subject=subject_name,
      observation="Human-readable fact with exact computed numbers.",
      method="method_name",
      evidence=(EvidenceRef(source_id, "observed_from", locator=subject),),
      assumptions=("Explicit statistical assumptions.",),
      limitations=("Explicit boundaries, e.g., correlation does not equal causation.",),
      result={"metric_name": value, ...},
      knowledge_state="statistical", # "observed" | "statistical" | "inferred"
  )
  ```

### Rules for New Methods

1. **No Data Modification:** Methods must only observe and profile; never mutate or clean the underlying DataFrame.
2. **Explainable Selection:** Any algorithm choice (e.g. Pearson vs. Spearman, IQR vs. MAD) must be captured in a `MethodSelection` rationale.
3. **Traceability:** Every finding must attach an `EvidenceRef` connecting back to the exact source and column/row locator.

---

## 3. Integrating the Explainer LLM as a `Synthesizer`

The Explainer LLM converts PARSE technical findings and cleaning proposals into verified plain-language summaries for operators and engineers.

### Core Synthesizer Protocol

In `parse/core/operations.py`, the `Synthesizer` protocol governs generation:

```python
@runtime_checkable
class Synthesizer(Protocol):
    def synthesize(
        self,
        evidence: RetrievalResult,
        analyses: Sequence[AnalysisResult] = (),
        **context: Any,
    ) -> SynthesisResult:
        """Combine evidence and analysis into a traceable result."""
        ...
```

### Explainer Integration Contract

The Explainer model is trained and evaluated on findings shaped as:
```python
{
    "finding_id": str,
    "source": str,        # "eda" | "cleaning_issue"
    "category": str,
    "kind": str,          # "observation" | "interpretation"
    "message": str,       # Source text containing numbers and facts
    "attributes": dict,   # Exact numerical values for fidelity checking
    "limitations": list,  # Limitations that must be preserved
}
```

### Architectural Adapter Pattern

To plug the Explainer in as a `Synthesizer`:

```python
class ExplainerSynthesizer:
    """Adapts the fine-tuned Explainer LLM to the Core Synthesizer protocol."""

    def __init__(self, model_loader_or_generator):
        self.generator = model_loader_or_generator

    def synthesize(
        self,
        evidence: RetrievalResult,
        analyses: Sequence[AnalysisResult] = (),
        **context: Any,
    ) -> SynthesisResult:
        # 1. Map AnalysisResult findings to Explainer finding dicts
        # 2. Invoke Explainer generator with prompt template
        # 3. Run automated fidelity checks (numerical_fidelity, causal_guard)
        # 4. Wrap explanation into SynthesisResult with provenance & citations
        ...
```

This integration preserves total separation: the Explainer LLM cannot alter the statistical ground truth, and its explanations remain audited by `eval/fidelity_checks.py`.
