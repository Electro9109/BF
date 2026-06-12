"""
prediction_pipeline.py
----------------------
High-level prediction pipeline.  Coordinates:
  1. Feature processing
  2. ML prediction (Ts / Tm / Tm-Ts)
  3. Similarity retrieval (stub — ready for ml/similarity.py)
  4. Output formatting for hybrid pipeline / RAG handoff

This is the module called by hybrid_pipeline.py.

Example
-------
  from pipeline.prediction_pipeline import PredictionPipeline

  pipe = PredictionPipeline()
  output = pipe.run(
      chemistry={...},
      test_condition="CO= 40% & N2=60%",
      burden="S1-70%+O1-30%",
      test_type="SO",
  )
  print(output["predictions"])
  print(output["summary"])
"""

from __future__ import annotations
import textwrap
from dataclasses import dataclass, field
from typing import Optional

from ml.predictor import Predictor
from ml.feature_processing import CHEM_COLS, TARGET_COLS


# ── Result dataclass ───────────────────────────────────────────────────────
@dataclass
class PredictionResult:
    predictions: dict              # {"Ts": float, "Tm": float, "Tm-Ts": float}
    inputs: dict                   # raw user inputs
    feature_vector: Optional[list] = None   # for similarity search
    similar_experiments: list = field(default_factory=list)   # stub
    summary: str = ""              # human-readable for RAG handoff

    def to_rag_query(self) -> str:
        """
        Build a natural-language query for the RAG system.
        Called by hybrid_pipeline.py to retrieve metallurgical theory.
        """
        ts  = self.predictions.get("Ts",    "N/A")
        tm  = self.predictions.get("Tm",    "N/A")
        tmt = self.predictions.get("Tm-Ts", "N/A")

        chem = self.inputs.get("chemistry", {})
        basicity = chem.get("Basicity", "N/A")
        feo      = chem.get("FeO %", "N/A")

        cond   = self.inputs.get("test_condition", "")
        burden = self.inputs.get("burden", "")

        return (
            f"Predicted softening temperature Ts={ts}°C, melting temperature Tm={tm}°C, "
            f"and melting interval Tm-Ts={tmt}°C for burden {burden} under {cond} conditions. "
            f"Basicity={basicity}, FeO={feo}%. "
            f"Why did these results occur? What metallurgical mechanisms explain this?"
        )


# ── Pipeline ───────────────────────────────────────────────────────────────
class PredictionPipeline:
    """
    Orchestrates the full prediction flow.

    Parameters
    ----------
    model_dir  : path to trained models directory (MLModels/)
    model_type : 'best', 'RandomForest', or 'XGBoost'
    targets    : which targets to predict (default: all three)
    """

    def __init__(
        self,
        model_dir: str = "MLModels",
        model_type: str = "best",
        targets: list = None,
    ):
        self.predictor = Predictor(
            model_dir=model_dir,
            model_type=model_type,
            targets=targets or TARGET_COLS,
        )
        # Similarity retriever will be injected here once ml/similarity.py is ready
        self._similarity_retriever = None

    def attach_similarity_retriever(self, retriever):
        """Attach the similarity module once ml/similarity.py is implemented."""
        self._similarity_retriever = retriever

    def run(
        self,
        chemistry: dict,
        test_condition: str = None,
        burden: str = None,
        test_type: str = None,
    ) -> PredictionResult:
        """
        Run the full prediction pipeline.

        Parameters
        ----------
        chemistry      : dict e.g. {"T Fe %": 57.6, "FeO %": 6.65, ...}
        test_condition : gas atmosphere string
        burden         : burden composition string
        test_type      : 'SO', 'SOP', or 'P'

        Returns
        -------
        PredictionResult
        """
        # ── Step 1: Predict ──────────────────────────────────────────────
        predictions = self.predictor.predict(
            chemistry=chemistry,
            test_condition=test_condition,
            burden=burden,
            test_type=test_type,
        )

        # ── Step 2: Similarity (stub) ────────────────────────────────────
        similar = []
        if self._similarity_retriever is not None:
            similar = self._similarity_retriever.find_similar(
                chemistry=chemistry,
                test_condition=test_condition,
                burden=burden,
                test_type=test_type,
                top_k=3,
            )

        # ── Step 3: Build result ─────────────────────────────────────────
        inputs = {
            "chemistry": chemistry,
            "test_condition": test_condition,
            "burden": burden,
            "test_type": test_type,
        }

        summary = self._build_summary(predictions, inputs, similar)

        return PredictionResult(
            predictions=predictions,
            inputs=inputs,
            similar_experiments=similar,
            summary=summary,
        )

    def _build_summary(
        self,
        predictions: dict,
        inputs: dict,
        similar: list,
    ) -> str:
        chem = inputs.get("chemistry", {})
        cond = inputs.get("test_condition") or "N/A"
        burden = inputs.get("burden") or "N/A"
        ttype  = inputs.get("test_type") or "N/A"

        lines = [
            "── Prediction Summary ──────────────────────────────",
            f"  Burden       : {burden}",
            f"  Test type    : {ttype}",
            f"  Atmosphere   : {cond}",
            "",
            "  Chemistry:",
        ]
        for col in CHEM_COLS:
            val = chem.get(col, "N/A")
            lines.append(f"    {col:<12}: {val}")

        lines += [
            "",
            "  Predictions:",
            f"    Ts     = {predictions.get('Ts',    'N/A')} °C",
            f"    Tm     = {predictions.get('Tm',    'N/A')} °C",
            f"    Tm-Ts  = {predictions.get('Tm-Ts', 'N/A')} °C",
        ]

        if similar:
            lines += ["", "  Similar Experiments:"]
            for exp in similar:
                lines.append(f"    - {exp}")

        lines.append("────────────────────────────────────────────────────")
        return "\n".join(lines)


# ── Standalone demo ────────────────────────────────────────────────────────
if __name__ == "__main__":
    pipe = PredictionPipeline()

    result = pipe.run(
        chemistry={
            "T Fe %": 57.62, "FeO %": 6.65, "SiO2 %": 4.23,
            "CaO %": 6.93, "Al2O3 %": 2.51, "MgO%": 1.89,
            "Basicity": 1.638298,
        },
        test_condition="CO= 40% & N2=60%",
        burden="S1-70%+O1-30%",
        test_type="SO",
    )

    print(result.summary)
    print("\nRAG Query:")
    print(result.to_rag_query())
