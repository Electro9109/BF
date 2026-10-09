# PARSE Qwen Explainer: Architecture, Deployment, & Fidelity Guide

The **PARSE Explainer** is a specialized natural language generation component for metallurgical and industrial data operations. Its sole responsibility is to translate structured analytical findings into clear, concise, and professional natural language.

---

## 1. Core Purpose and Architectural Boundary

### Strict Responsibility
The Explainer has **exactly one responsibility**:
> **Translate analyser findings and supplied recommendations into concise, understandable natural language.**

### Absolute Architectural Invariants
1. **Never Calculate or Analyse:** The Explainer does not compute statistics, calculate metrics, or verify mathematics.
2. **Never Predict Outcomes:** It does not forecast future states or make predictive estimates.
3. **Never Invent Recommendations:** Recommendations are only included if already produced by the upstream Analysis Engine. The Explainer never invents or synthesizes actions.
4. **Never Retrieve Documents:** It does not query RAG stores or external vector databases.
5. **Deterministic Source of Truth:** The **Analysis Engine** (`parse.analysis`) remains the sole, unquestioned source of truth.

```
┌─────────────────────────────────┐
│     Analysis Engine (Truth)     │
│   (Orchestrator, EDA, Cleaning) │
└────────────────┬────────────────┘
                 │ Finding / EDAResult
                 ▼
┌─────────────────────────────────┐
│    Explainer Adapter (v1.0)     │
│   (parse.explainer_adapter)     │
└────────────────┬────────────────┘
                 │ Explainer Record Dict
                 ▼
┌─────────────────────────────────┐
│       Explainer Pipeline        │
│      (parse.explainer_llm)      │
│  - Prompt Construction          │
│  - Greedy Local Generation      │
│  - Token-Slice Echo Protection  │
└────────────────┬────────────────┘
                 │ Generated Text
                 ▼
┌─────────────────────────────────┐
│         Fidelity Gate           │
│     (eval.fidelity_checks)      │
│  - Numerical Fidelity           │
│  - Causal Language              │
│  - Limitation Preservation      │
└────────┬───────────────┬────────┘
     Pass│          Fail │
         ▼               ▼
┌────────────────┐ ┌───────────────┐
│  Return Text   │ │ Safe Fallback │
│ (ExplainerRes) │ │ (finding.msg) │
└────────────────┘ └───────────────┘
```

---

## 2. Model & Adapter Specifications

- **Base Model:** `Qwen/Qwen2.5-3B-Instruct`
- **Adapter Version:** `v0.3` (`parse/models/qwen3-explainer-v0.3`)
- **Adapter Architecture:** LoRA (Low-Rank Adaptation)
  - Rank ($r$): `16`
  - Alpha ($\alpha$): `32`
  - Target Modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- **Tokenizer:** Ships with adapter config (`tokenizer.json`, `tokenizer_config.json`); automatically preferred when present.

---

## 3. Environment Variables & Runtime Configuration

| Variable | Type | Default | Description |
|---|---|---|---|
| `PARSE_EXPLAINER_ENABLED` | String/Bool | `"1"` (Enabled) | Set to `"0"`, `"false"`, or `"off"` to disable model loading and force graceful deterministic fallback. |
| `PARSE_BASE_MODEL_PATH` | Directory Path | Empty | Path to a local directory containing pre-downloaded `Qwen2.5-3B-Instruct` base model weights for air-gapped or offline environments. |

---

## 4. Prompt Engineering & Anti-Hallucination Guardrails

Prompt construction is centralized in `parse/explainer_prompt.py`:

1. **System Prompt:**
   - Enforces strict constraints: never invent numbers, never introduce causal claims for observational correlations, never extrapolate beyond the provided finding.
2. **User Turn Structure:**
   - Clearly presents finding category, kind, raw message, attributes (exact numbers), and limitations.
   - If a recommendation exists, it is presented with its exact strength (`must`, `should`, `consider`).
3. **Prompt-Echo Protection:**
   - Decodes strictly from `output_ids[0][n_prompt:]`, slicing off all input prompt tokens. The model output is never polluted by repeated prompt text.

---

## 5. Inference Constraints

Inference uses strictly deterministic decoding in `parse/explainer_llm.py`:
- `do_sample=False`
- `temperature=0.0`
- `max_new_tokens=256`
- Memory-efficient execution with `torch.no_grad()`.

---

## 6. Fidelity Gates & Advisory Checks

Every generated explanation is evaluated by `eval/fidelity_checks.py`:

### Hard Gates (Failures Trigger Immediate Fallback)
1. **Numerical Fidelity (`numerical_fidelity`):**
   - Asserts that every number present in the source finding or numeric attributes appears accurately in the output. No numbers may be altered or dropped.
2. **Causal Language Gate (`causal_language_check`):**
   - Flags forbidden causal verbs (`causes`, `drives`, `leads to`, `results in`) if the source finding is observational or associational rather than an established causal mechanism.
3. **Limitation Preservation (`limitation_preserved`):**
   - Ensures any statistical limitation stated in the source finding (e.g., *"A statistical outlier is not automatically a data error"*) is faithfully preserved in the explanation.

### Soft Advisory Gate (Auditing Only)
4. **Unsupported Novelty (`unsupported_novelty`):**
   - Evaluates the proportion of novel terms (words not found in the source finding, attributes, or domain stopwords).
   - Serves as an advisory human-review warning logged via `logger.warning`.
   - **Does NOT trigger fallback** to avoid rejecting legitimate metallurgical rephrasing and terminology.

---

## 7. Zero-Downtime Fallback Contract

If:
- `PARSE_EXPLAINER_ENABLED=0`
- Adapter weights are missing
- CUDA runs out of memory or PyTorch throws an exception
- Any hard fidelity gate fails

The system immediately and silently falls back:
- `fallback_used = True`
- `text = record["message"]` (deterministic raw finding message)
- `validation_detail = reason`

The calling application receives a valid `ExplainerResult` without interruption.

---

## 8. Deployment & Hardware Recommendations

### System Requirements
- **CPU Deployment:** 8GB+ RAM. Inference latency ~1.5s–3s per finding on modern multi-core x86_64 CPUs.
- **GPU Deployment (Recommended for High Throughput):** NVIDIA GPU with 6GB+ VRAM (e.g., RTX 3060/4060, T4, A10G). Supports `torch.bfloat16` or `torch.float16`.
- **Quantization:** Compatible with `bitsandbytes` 4-bit / 8-bit loading for low-resource environments.

### Air-Gapped Setup
1. Download `Qwen/Qwen2.5-3B-Instruct` to a local folder `/models/qwen2.5-3b-instruct`.
2. Set `export PARSE_BASE_MODEL_PATH=/models/qwen2.5-3b-instruct`.
3. The adapter is self-contained within `parse/models/qwen3-explainer-v0.3`. No internet access is required.
