"""
ml/condition_parser.py
----------------------
Hybrid parser (Regex + LLM fallback) that extracts numerical experiment
conditions and burden composition from a free-text description.

Phase-2 enhancements
--------------------
* Auto-normalises atmosphere (CO+H2+N2) to sum to 100 when close but not exact.
* Auto-normalises burden (sinter+ore+pellet+other) likewise.
* Returns a `warnings` list for:
    - atmosphere total deviating from 100 by more than 5 %.
    - burden total deviating from 100 by more than 5 %.
    - any single gas/burden component outside [0, 100].
    - CO below physical minimum (< 30 %) — model extrapolation risk.
* If LLM JSON is malformed but CO/H2/N2 were found by regex the function
  still returns a best-effort result instead of all zeros.
"""

from __future__ import annotations
import json
import re
from typing import Optional


# ── Prompt fragments ──────────────────────────────────────────────────────────

SYSTEM_INSTRUCTION = (
    "You are an expert data extractor for a metallurgical blast furnace model. "
    "Your task is to read a description of an experiment's atmosphere and burden composition, "
    "and output a strictly formatted JSON object extracting the percentages."
)

FEW_SHOT_PROMPT = """\
You must output a JSON object with EXACTLY these keys:
{{
  "CO_pct": <float>,
  "H2_pct": <float>,
  "N2_pct": <float>,
  "sinter_pct": <float>,
  "ore_pct": <float>,
  "pellet_pct": <float>,
  "other_pct": <float>,
  "test_type": "<SO|SOP|P>"
}}

Rules:
1. All atmosphere _pct values must sum to 100 (CO + H2 + N2 = 100). If not specified, assume N2 is the balance.
2. All burden _pct values must sum to 100 (sinter + ore + pellet + other = 100).
3. 'test_type' must be 'SO' (Sinter+Ore), 'SOP' (Sinter+Ore+Pellet), or 'P' (Pellet) depending on the burden.
4. Output ONLY valid JSON, with no markdown formatting, no code blocks, and no extra text.

Examples:
Input: "Atmosphere has 40% CO and no hydrogen, rest is N2. Burden is mostly S1 sinter at 70%, with 30% O1 ore."
Output: {{"CO_pct": 40.0, "H2_pct": 0.0, "N2_pct": 60.0, "sinter_pct": 70.0, "ore_pct": 30.0, "pellet_pct": 0.0, "other_pct": 0.0, "test_type": "SO"}}

Input: "We ran a test with 36% CO, 4% H2, and 60% N2. The burden was 65% sinter, 25% ore, and 10% pellet."
Output: {{"CO_pct": 36.0, "H2_pct": 4.0, "N2_pct": 60.0, "sinter_pct": 65.0, "ore_pct": 25.0, "pellet_pct": 10.0, "other_pct": 0.0, "test_type": "SOP"}}

Input: "100% pellet burden with 34% CO and 6% hydrogen, 60% nitrogen."
Output: {{"CO_pct": 34.0, "H2_pct": 6.0, "N2_pct": 60.0, "sinter_pct": 0.0, "ore_pct": 0.0, "pellet_pct": 100.0, "other_pct": 0.0, "test_type": "P"}}

Now process the following input.

Input: "{text}"
Output:"""


# ── Internal helpers ──────────────────────────────────────────────────────────

def _normalise_group(vals: list[float], labels: list[str], warnings: list[str],
                     group_name: str) -> list[float]:
    """
    Normalise a list of percentages to sum to 100 if their total is within
    [85, 115].  Appends a warning string if adjustment was made or if the
    total is outside that window.
    """
    total = sum(vals)
    if abs(total - 100.0) < 1e-3:           # already exactly 100
        return vals
    if total == 0.0:                         # nothing to normalise
        return vals

    deviation = abs(total - 100.0)
    if deviation <= 5.0:
        # Silent normalisation — small floating-point drift
        factor = 100.0 / total
        return [v * factor for v in vals]
    elif deviation <= 15.0:
        # Normalise with a warning
        factor = 100.0 / total
        normed = [v * factor for v in vals]
        warnings.append(
            f"{group_name} percentages summed to {total:.1f} % "
            f"(expected 100 %). Auto-normalised to 100 %."
        )
        return normed
    else:
        # Large deviation — normalise but warn more urgently
        factor = 100.0 / total
        normed = [v * factor for v in vals]
        warnings.append(
            f"⚠ {group_name} percentages summed to {total:.1f} %, which is "
            f"far from 100 %. Check your input — values were auto-normalised."
        )
        return normed


def _check_bounds(name: str, val: float, lo: float, hi: float,
                  warnings: list[str]) -> None:
    if val < lo or val > hi:
        warnings.append(
            f"⚠ {name}={val:.1f} is outside the expected range [{lo}, {hi}]."
        )


def _infer_test_type(sinter: float, ore: float, pellet: float) -> str:
    if pellet > 0 and sinter == 0 and ore == 0:
        return "P"
    if sinter > 0 and ore > 0 and pellet > 0:
        return "SOP"
    return "SO"


def _build_result(co, h2, n2, sinter, ore, pellet, other,
                  test_type: str, warnings: list[str]) -> dict:
    """Normalise, run sanity checks, and build the result dict."""

    # ── Atmosphere normalisation ──
    atm_vals = _normalise_group(
        [co, h2, n2], ["CO_pct", "H2_pct", "N2_pct"], warnings, "Atmosphere"
    )
    co, h2, n2 = atm_vals

    # ── Burden normalisation ──
    burden_vals = _normalise_group(
        [sinter, ore, pellet, other],
        ["sinter_pct", "ore_pct", "pellet_pct", "other_pct"],
        warnings, "Burden"
    )
    sinter, ore, pellet, other = burden_vals

    # ── Individual component sanity checks ──
    for label, val in [("CO_pct", co), ("H2_pct", h2), ("N2_pct", n2),
                       ("sinter_pct", sinter), ("ore_pct", ore),
                       ("pellet_pct", pellet), ("other_pct", other)]:
        _check_bounds(label, val, 0.0, 100.0, warnings)

    # ── Domain-specific warning: very low CO ──
    if co < 30.0 and (co + h2 + n2) > 0:
        warnings.append(
            f"CO_pct={co:.1f} % is below 30 %. Most training data has CO >= 30 %; "
            "predictions may be less reliable."
        )

    num_components = sum(1 for v in [sinter, ore, pellet, other] if v > 0)

    return {
        "CO_pct":          round(co, 2),
        "H2_pct":          round(h2, 2),
        "N2_pct":          round(n2, 2),
        "sinter_pct":      round(sinter, 2),
        "ore_pct":         round(ore, 2),
        "pellet_pct":      round(pellet, 2),
        "other_pct":       round(other, 2),
        "num_components":  num_components,
        "test_type":       test_type,
        "warnings":        warnings,
    }


# ── Regex extraction ──────────────────────────────────────────────────────────

def _regex_extract(text_lower: str) -> dict:
    """
    Extract numeric gas + burden percentages from text using regex.
    Returns a dict with the extracted floats (all default 0.0 if not found).
    """
    co = h2 = n2 = 0.0
    sinter = ore = pellet = other = 0.0

    # ── Gas ──
    for token, pattern_a, pattern_b in [
        ("co",  r'(?:co[=:\s]+|carbon monoxide\s*(?:is|at)?\s*)(\d+(?:\.\d+)?)%',
                r'(\d+(?:\.\d+)?)\s*%\s*co'),
        ("h2",  r'(?:h2[=:\s]+|hydrogen\s*(?:is|at)?\s*)(\d+(?:\.\d+)?)%',
                r'(\d+(?:\.\d+)?)\s*%\s*h2'),
        ("n2",  r'(?:n2[=:\s]+|nitrogen\s*(?:is|at)?\s*)(\d+(?:\.\d+)?)%',
                r'(\d+(?:\.\d+)?)\s*%\s*n2'),
    ]:
        m = re.search(pattern_a, text_lower) or re.search(pattern_b, text_lower)
        if m:
            if token == "co":
                co = float(m.group(1))
            elif token == "h2":
                h2 = float(m.group(1))
            elif token == "n2":
                n2 = float(m.group(1))

    # Infer N2 as balance if not mentioned
    if n2 == 0.0 and (co > 0.0 or h2 > 0.0) and (co + h2 < 100.0):
        n2 = 100.0 - co - h2

    # ── Burden shortcuts ──
    shortcuts = {
        "sinter": ["100% sinter", "completely sinter", "sinter only"],
        "ore":    ["100% ore",    "completely ore",    "ore only"],
        "pellet": ["100% pellet", "completely pellet", "pellet only"],
    }
    for material, phrases in shortcuts.items():
        if any(p in text_lower for p in phrases):
            if material == "sinter":
                sinter = 100.0
            elif material == "ore":
                ore = 100.0
            elif material == "pellet":
                pellet = 100.0

    # ── Burden percentages ──
    for token, pattern_a, pattern_b in [
        ("sinter",  r'(?:sinter[s\s]*(?:at)?\s*)(\d+(?:\.\d+)?)%',
                    r'(\d+(?:\.\d+)?)\s*%\s*sinter'),
        ("ore",     r'(?:ore[s\s]*(?:at)?\s*)(\d+(?:\.\d+)?)%',
                    r'(\d+(?:\.\d+)?)\s*%\s*ore'),
        ("pellet",  r'(?:pellet[s\s]*(?:at)?\s*)(\d+(?:\.\d+)?)%',
                    r'(\d+(?:\.\d+)?)\s*%\s*pellet'),
    ]:
        m = re.search(pattern_a, text_lower) or re.search(pattern_b, text_lower)
        if m:
            val = float(m.group(1))
            if token == "sinter" and sinter == 0.0:
                sinter = val
            elif token == "ore" and ore == 0.0:
                ore = val
            elif token == "pellet" and pellet == 0.0:
                pellet = val

    # Single material mentioned with no percentage → assume 100 %
    mentions = [w for w in ["sinter", "ore", "pellet"] if w in text_lower]
    if sinter + ore + pellet == 0.0 and len(mentions) == 1:
        if mentions[0] == "sinter":
            sinter = 100.0
        elif mentions[0] == "ore":
            ore = 100.0
        elif mentions[0] == "pellet":
            pellet = 100.0

    return dict(co=co, h2=h2, n2=n2,
                sinter=sinter, ore=ore, pellet=pellet, other=other)


# ── Public API ─────────────────────────────────────────────────────────────────

def parse_condition_nl(text: str, tokenizer, model,
                       require_llm: bool = False) -> dict:
    """
    Use a hybrid parser to extract numeric conditions from a free-text description.

    1. Regex extraction  — fast, deterministic.
    2. LLM fallback      — used when regex yields no atmosphere OR no burden.

    Returns a dict with keys:
        CO_pct, H2_pct, N2_pct,
        sinter_pct, ore_pct, pellet_pct, other_pct,
        num_components, test_type, warnings (list[str])
    """
    text_lower = text.lower()
    warnings: list[str] = []

    # ── Step 1: Regex ────────────────────────────────────────────────────────
    extracted = _regex_extract(text_lower)
    co, h2, n2 = extracted["co"], extracted["h2"], extracted["n2"]
    sinter, ore = extracted["sinter"], extracted["ore"]
    pellet, other = extracted["pellet"], extracted["other"]

    has_atmosphere = (co > 0.0 or h2 > 0.0 or n2 > 0.0)
    has_burden     = (sinter > 0.0 or ore > 0.0 or pellet > 0.0)

    if has_atmosphere and has_burden and not require_llm:
        test_type = _infer_test_type(sinter, ore, pellet)
        return _build_result(co, h2, n2, sinter, ore, pellet, other,
                             test_type, warnings)

    # ── Step 2: LLM fallback ─────────────────────────────────────────────────
    if tokenizer is None or model is None:
        warnings.append(
            "Natural-language parsing was incomplete and no local LLM was supplied; "
            "regex values were retained."
        )
        return _build_result(
            co, h2, n2, sinter, ore, pellet, other,
            _infer_test_type(sinter, ore, pellet), warnings,
        )

    import torch

    prompt = FEW_SHOT_PROMPT.format(text=text)
    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user",   "content": prompt},
    ]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([formatted_prompt], return_tensors="pt", padding=True)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=150,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][input_len:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # Strip markdown code fences if present
    response = re.sub(r'```json\s*', '', response)
    response = re.sub(r'```\s*',     '', response)
    response = response.strip()

    try:
        parsed = json.loads(response)
        co      = float(parsed.get("CO_pct",     co))
        h2      = float(parsed.get("H2_pct",     h2))
        n2      = float(parsed.get("N2_pct",     n2))
        sinter  = float(parsed.get("sinter_pct", sinter))
        ore     = float(parsed.get("ore_pct",    ore))
        pellet  = float(parsed.get("pellet_pct", pellet))
        other   = float(parsed.get("other_pct",  other))
        test_type = parsed.get("test_type", _infer_test_type(sinter, ore, pellet))
    except (json.JSONDecodeError, ValueError):
        warnings.append(
            "LLM output could not be parsed as JSON — falling back to regex values."
        )
        test_type = _infer_test_type(sinter, ore, pellet)

    return _build_result(co, h2, n2, sinter, ore, pellet, other,
                         test_type, warnings)
