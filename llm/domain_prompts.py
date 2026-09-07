"""BF/metallurgical synthesis policy kept outside LLM infrastructure."""

from config.retrieval import COMPARE_KEYWORDS, EFFECT_KEYWORDS, PROCESS_KEYWORDS

SYSTEM_INSTRUCTION = (
    "You are a metallurgical process assistant. "
    "Answer questions using ONLY the provided Context. "
    "Do not invent facts or use outside knowledge. "
    "Structure your answer clearly. If the answer is not in the Context, say so."
)


def build_domain_instructions(question: str) -> str:
    """Choose the BF-specific answer structure for a query."""
    query = question.lower()
    if any(keyword in query for keyword in PROCESS_KEYWORDS):
        return (
            "1. Give a concise definition (1-2 sentences).\n"
            "2. Explain the key steps or mechanisms in order.\n"
            "3. Mention important temperatures, compositions, or conditions.\n"
            "4. State relevance to blast furnace or ironmaking operation.\n"
            "Do NOT copy raw text from the context. Synthesize it."
        )
    if any(keyword in query for keyword in EFFECT_KEYWORDS):
        return (
            "1. State what is affected and how.\n"
            "2. Explain the metallurgical mechanism behind the effect.\n"
            "3. Give numerical values or ranges mentioned in the context.\n"
            "Do NOT copy raw text from the context. Synthesize it."
        )
    if any(keyword in query for keyword in COMPARE_KEYWORDS):
        return (
            "1. Identify key differences between the items.\n"
            "2. State which conditions favour each option.\n"
            "3. Summarize trade-offs mentioned in the context.\n"
            "Do NOT copy raw text from the context. Synthesize it."
        )
    return (
        "1. Answer directly and concisely.\n"
        "2. Support the answer with specific details from the context.\n"
        "3. Do NOT copy raw text. Synthesize the information."
    )