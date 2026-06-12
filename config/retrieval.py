"""
config/retrieval.py
────────────────────
Retrieval tuning knobs: result counts, topic routing keywords, and
section-based re-rank weights.
"""

# Number of chunks returned by RetrievalEngine.search() by default
TOP_K = 5

# Minimum candidate chunks required for a topic filter to be considered
# "valid". If a topic match returns fewer than this, fall back to the
# full corpus (see retrieval/retriever.py).
MIN_TOPIC_CANDIDATES = 3

# Section headings whose content adds no retrieval value.
# Chunks whose first words match any of these are silently dropped from the index.
JUNK_HEADINGS = {
    "references", "citations", "sources", "bibliography",
    "further reading", "see also", "notes", "acknowledgements",
    "acknowledgments", "appendix",
}

# Sections kept in the index but penalised for explanatory queries
LOW_VALUE_SECTIONS = {"limitations", "disadvantages", "challenges", "drawbacks"}

# Sections boosted for explanatory queries ("explain", "how", "what is", …)
EXPLANATORY_SECTIONS = {
    "overview", "definition", "process", "key mechanisms", "mechanism",
    "production", "microstructure", "blast furnace significance",
    "key takeaways", "summary", "effect", "effects",
}

# Re-rank multipliers
EXPLANATORY_BOOST = 1.15
LOW_VALUE_PENALTY = 0.75

# Topic keyword map: canonical topic name -> query substrings that signal it.
# Longer / more specific phrases are checked first inside _detect_topic so
# "cohesive zone" wins over "cohesive".
TOPIC_KEYWORDS: dict[str, list[str]] = {
    "sinter":              ["sinter", "sintering", "coke breeze", "sfca", "traveling grate"],
    "pellet":              ["pellet", "pellets", "pelletizing", "induration", "balling"],
    "basicity":            ["basicity", "cao/sio2", "binary basicity", "flux", "fluxing"],
    "cohesive_zone":       ["cohesive zone", "cohesive", "softening zone"],
    "softening_melting":   ["softening", "melting", "tm-ts", "tm_ts"],
    "hydrogen_reduction":  ["hydrogen", "h2", "h2 reduction", "hydrogen reduction"],
    "reducibility":        ["reducibility", "reducible", "reduction degree", "rdi"],
    "pressure_drop":       ["pressure drop", "permeability", "delp", "del p", "ndm"],
    "slag_formation":      ["slag", "slag formation", "viscosity", "liquidus"],
    "burden_distribution": ["burden", "burden distribution", "charging", "stock line"],
}

# Query-type keyword sets (used by reranker.py and llm/prompts.py)
PROCESS_KEYWORDS = {"explain", "how", "process", "mechanism", "describe", "what is", "what are"}
EFFECT_KEYWORDS = {"effect", "affect", "impact", "influence", "why", "reason", "cause"}
COMPARE_KEYWORDS = {"compare", "difference", "versus", "vs", "better", "worse"}
EXPLANATORY_QUERY_KEYWORDS = {
    "explain", "what is", "what are", "how does", "describe",
    "define", "overview", "process", "mechanism",
}
