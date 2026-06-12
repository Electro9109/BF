"""
llm/prompts.py - Query-type detection and structured prompt templates.

Small models (Qwen-0.5B) respond significantly better to numbered
step-by-step instructions than to open-ended 'answer using context'
prompts. Each template routes the model toward a different answer
structure appropriate for the query type.
"""

from config.retrieval import COMPARE_KEYWORDS, EFFECT_KEYWORDS, PROCESS_KEYWORDS

SYSTEM_INSTRUCTION = (
    "You are a metallurgical process assistant. "
    "Answer questions using ONLY the provided Context. "
    "Do not invent facts or use outside knowledge. "
    "Structure your answer clearly. If the answer is not in the Context, say so."
)


def build_prompt(question: str, context: str) -> str:
    """Select a structured prompt template based on query type."""
    q = question.lower()

    if any(kw in q for kw in PROCESS_KEYWORDS):
        instructions = (
            "1. Give a concise definition (1-2 sentences).\n"
            "2. Explain the key steps or mechanisms in order.\n"
            "3. Mention any important temperatures, compositions, or conditions.\n"
            "4. State the relevance to blast furnace or ironmaking operation.\n"
            "Do NOT copy raw text from the context. Synthesize it."
        )
    elif any(kw in q for kw in EFFECT_KEYWORDS):
        instructions = (
            "1. State what is being affected and how (increase / decrease / improve / worsen).\n"
            "2. Explain the metallurgical mechanism behind this effect.\n"
            "3. Give any numerical values or ranges mentioned in the context.\n"
            "Do NOT copy raw text from the context. Synthesize it."
        )
    elif any(kw in q for kw in COMPARE_KEYWORDS):
        instructions = (
            "1. Identify the key differences between the two items.\n"
            "2. State which conditions favour each option.\n"
            "3. Summarize any trade-offs mentioned in the context.\n"
            "Do NOT copy raw text from the context. Synthesize it."
        )
    else:
        instructions = (
            "1. Answer directly and concisely.\n"
            "2. Support your answer with specific details from the context.\n"
            "3. Do NOT copy raw text. Synthesize the information."
        )

    return (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Instructions:\n{instructions}"
    )
