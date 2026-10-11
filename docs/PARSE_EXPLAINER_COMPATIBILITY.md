# PARSE Explainer v0.3 Compatibility Investigation

**Checkout:** `master`, latest commit `0e84169` (`Polish explainer fallback and validation contract`); working tree was clean at inspection.

## 1. Training Contract

The original v0.3 training notebook/script, source or processed dataset, split files, preprocessing code, and target examples were not found in the repository or the recursive local artifact search. `lora/README.md` is an unfilled generic model card; `lora/v02_heldout_baseline.txt` contains v0.2 examples only. The project-generated `eval/legacy_eval_set.json` and `eval/parse_eval_set.json` are evaluation fixtures, not evidence of v0.3 training records.

Therefore, v0.3's actual input serialization, prompt roles and wording, training chat-template use, truncation/max sequence length, target style, recommendation coverage, and held-out performance are **Unknown**. They cannot be reconstructed from the current inference prompt or model name.

Verified from the physical `parse/models/qwen3-explainer-v0.3/adapter_config.json` only:

- Base reference: `Qwen/Qwen3-4B-Instruct-2507`; revision is `null`.
- PEFT type/task: `LORA` / `CAUSAL_LM`; PEFT version recorded as `0.21.2`.
- Rank 16, alpha 32, dropout 0.05, bias `none`, `lora_bias=false`.
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj` only; `modules_to_save=null`; no saved embedding/module extension is declared.
- Adapter weights exist locally (`adapter_model.safetensors`, 47,224,624 bytes). Their presence does not establish training behavior.

The adapter-local tokenizer config says `Qwen2Tokenizer`, `model_max_length=1010000`, pad `<|endoftext|>`, EOS `<|im_end|>`, and has no `chat_template`. In the project virtualenv (`transformers 4.57.6`) `AutoTokenizer.from_pretrained(adapter_dir, local_files_only=True)` fails before inference: `extra_special_tokens` is a list, but this installed tokenizer implementation treats it as a mapping. The root-level renamed tokenizer config contains a template, but its provenance and use during v0.3 training are unknown.

## 2. Live Input Contract

The Data Explorer selects one `Finding` from `eda_result.findings` in `app_web.py`, then calls `ExplainerSynthesizer.synthesize_finding(selected_finding, recommendation=...)`. That method wraps only that finding, and the synthesizer converts its one-item `findings` list using `finding_to_explainer_example`; it does not send the entire analysis bundle. The adapter maps the canonical finding ID/category, uses `finding.observation` as the message (with selected finding-type-specific renderings), copies `finding.result` to `attributes`, adds subject-derived columns where relevant, and copies limitations. Recommendation remains a separate optional string.

The prompt builder creates two messages: a system instruction demanding one plain-language paragraph and a user message with finding message, category, kind, attributes, limitations, and an optional recommendation section. `explainer_llm._generate_raw` calls `tokenizer.apply_chat_template(..., tokenize=False, add_generation_prompt=True)`, tokenizes that string, and slices generated tokens after the prompt. The exact final token string is not produced in this environment because loading the configured adapter tokenizer fails as described above.

Example live-shaped record below is from the checked-in **project-generated eval fixture**, not represented as an original training record:

```json
{
  "schema_version": "1.0",
  "finding_id": "outliers_measurement",
  "source": "eda",
  "category": "statistical",
  "kind": "observation",
  "message": "Column 'measurement' contains 3 IQR-based statistical outlier(s).",
  "attributes": {"count": 3, "column": "measurement"},
  "limitations": ["Unusualness is not evidence of error."]
}
```

Current prompt user-turn representation for that record (without a separately supplied recommendation):

```text
Finding:
  message: Column 'measurement' contains 3 IQR-based statistical outlier(s).
  category: statistical
  kind: observation
  attributes: count=3, column='measurement'
  limitations: Unusualness is not evidence of error.
```

No actual model response or evaluator result exists for this investigation because generation did not run.

## 3. Compatibility Matrix

| Aspect | Verified v0.3 training format | Live PARSE format | Compatible? | Evidence / severity / required action |
|---|---|---|---|---|
| Finding representation | Unknown | One adapter record per selected canonical `Finding`; fields include message, attributes, limitations | Unknown | Training records absent. Obtain source dataset/pipeline before designing an asserted training-format adapter. |
| Field names | Unknown | `schema_version`, `finding_id`, `source`, `category`, `kind`, `message`, `attributes`, `limitations` | Unknown | The names in `docs/EXTENDING.md` are project contract documentation, not verified training evidence. |
| Numeric values and units | Unknown | Original result attributes plus rendered message; adapter has some type-specific rounding/formatting | Unknown | Need compare actual training examples to determine acceptable precision and units. |
| Recommendations | Unknown | Separate optional text supplied from UI; not part of canonical finding record | Unknown | Whether v0.3 saw recommendation examples is unknown. The runtime evaluator now receives the supplied recommendation. |
| System/user messages | Unknown | Current builder emits a custom strict-translation system prompt and structured-text user prompt | Unknown | No prompt correction is justified without original training prompt. |
| Chat template/tokenizer | Unknown | Loader prefers adapter tokenizer; installed Transformers fails loading it; config has no template | No for current runtime | **Confirmed runtime blocker.** Verify tokenizer compatibility and template provenance, then run real load/generation. |
| Expected output | Unknown | Current prompt requests one plain-language paragraph; fidelity gates inspect numeric, causal, limitation, recommendation, and metric semantics | Unknown | Training target format is unavailable; do not claim target alignment. |
| Sequence length | Unknown | `MAX_NEW_TOKENS=256`; adapter tokenizer claims max length 1,010,000 | Unknown | Training maximum/truncation unavailable; live prompt length not empirically compared. |
| Selected finding identity | Unknown | One selected finding is adapted and forwarded; evaluator now uses the exact supplied record | Path verified | Regression tests verify exact-record and missing-source behavior. |

## 4. Root Cause

**Confirmed:** The final UI evaluation call omitted the separately supplied recommendation, so recommendation fidelity was not checked against the recommendation shown to generation. The evaluator also returned `passed=True, evaluated=False` when no source findings existed, and filtered a supplied selected record by ID instead of checking that exact record. These are corrected in this change.

**Confirmed runtime blocker:** In the project virtualenv, adapter-tokenizer loading fails on its `extra_special_tokens` shape under Transformers 4.57.6. The installed runtime has PEFT 0.21.2 and Torch 2.14.1+cpu. No Hugging Face cache for the configured base model is present and no `PARSE_*`/`HF_*` model path is configured. Root-level Qwen3 configuration and sharded model files exist, but are not configured as the inference base path and do not have standard tokenizer filenames.

**Unknown, not a confirmed mismatch:** Whether the adapter was trained on materially different fields, prompt messages, recommendation-aware tasks, or output constraints. Fluency, fixture success, adapter loadability in another environment, and current docs do not answer those questions.

## 5. Decision

Do not retrain and do not add a purported training-format conversion yet. The available evidence cannot determine whether the live record needs a template or whether the desired recommendation-aware task was represented in training. Preserve the current adapter and prompt. First obtain the original v0.3 training source/data/representative sanitized records and establish a supported base/tokenizer runtime; then compare examples and run a real baseline plus converted live findings. Current code changes are limited to fixing independently verified fidelity-evaluation defects.

## 6. Files Changed

- `parse/fidelity_evaluator.py`: use the exact selected record, pass the supplied recommendation into checks, and mark unevaluable results failed/fallback.
- `app_web.py`: pass the same recommendation used for generation into final evaluation.
- `tests/test_explainer_integration.py`: cover recommendation checking, exact selected-record checking, and fail-closed missing-source behavior.
- `docs/EXPLAINER.md`: correct base model and LoRA target-module documentation from the local adapter config.
- `docs/PARSE_EXPLAINER_COMPATIBILITY.md`: this evidence report.

No adapter, adapter weights, base-model weights, prompt text, analysis module, or UI design was changed.

## 7. Tests

Focused commands:

- `.venv\\Scripts\\python.exe -m pytest tests/test_explainer_integration.py -q` — **12 passed**.
- `.venv\\Scripts\\python.exe -m pytest tests/test_explainer_adapter.py tests/test_explainer_integration.py tests/test_explainer_llm.py tests/test_explainer_prompt.py tests/test_fidelity_checks.py -q` — **90 passed**.
- `.venv\\Scripts\\python.exe -m pytest -q` — **248 passed**.

The focused tests mock generation or exercise deterministic fallback/evaluator behavior; they are not model inference tests.

## 8. Real Model Evidence

- Adapter directory and adapter weights: present.
- Adapter metadata references the required Qwen3-4B-Instruct-2507 base and PEFT LoRA.
- PEFT is installed in `.venv`; the standalone system Python environment lacks it.
- Actual adapter tokenizer load: **failed** under the project's `.venv` Transformers 4.57.6 due to `extra_special_tokens` metadata shape.
- Configured base cache/path: **not available** (`HF` cache has no configured model snapshot; no path environment variables set). Root-level Qwen3 configuration/shards exist but are not selected by the loader.
- Adapter attachment: **not reached**.
- Generation and training/live baselines: **not run**.

## 9. Green-Zone Checklist

- **A — Training contract understood:** Blocked. Original training source, examples, prompt construction, and split are unavailable.
- **B — Input alignment:** Partial. Live selected-finding path and current serialization are traced; alignment to v0.3 cannot be established without training examples.
- **C — Real inference:** Blocked. Adapter tokenizer loading fails and configured base model is unavailable to the application; no generation claim is made.
- **D — Fidelity and application behavior:** Partial. Selected-record and recommendation checks, fail-closed evaluation, and mocked tests are addressed; full suite and actual UI/model path remain to validate.
- **E — Decision justified:** Passed for this diagnostic state: no retraining or speculative conversion; resolve runtime compatibility and obtain source training evidence before choosing a template or fine-tuning.

## 10. Remaining Work

1. Supply the v0.3 training notebook/script, exact dataset examples/targets, splits, tokenizer/chat template, and preprocessing/truncation settings.
2. Configure a supported local or cached Qwen3-4B-Instruct-2507 base and resolve adapter-tokenizer compatibility without modifying v0.3 weights.
3. Run real adapter attachment and generation on several training-format examples and representative live PARSE findings; retain sanitized inputs, outputs, and hard-check results.
4. Build the controlled compatibility set and update the decision only from those observations.
5. Run the focused and full tests, then record their final results.