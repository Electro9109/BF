"""Generic prompt formatting for context-grounded generation."""


def build_prompt(question: str, context: str, instructions: str = "") -> str:
    """Combine evidence, question, and caller-provided synthesis policy."""
    return (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Instructions:\n{instructions or 'Answer using only the provided context.'}"
    )
