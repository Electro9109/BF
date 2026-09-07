"""Command-line entry point for the current offline RAG workflow.

Use ``streamlit run app_web.py`` for the full interface.
"""

import argparse

from config.paths import DOCS_DIR, LLM_MODEL_DIR
from config.retrieval import TOP_K
from pipeline.rag_pipeline import answer_question, build_engine, load_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the local metallurgical RAG system.")
    parser.add_argument("question", help="Question to ask about the indexed documents")
    args = parser.parse_args()

    engine, _ = build_engine(DOCS_DIR)
    if engine is None:
        raise SystemExit(f"No .txt documents found in {DOCS_DIR}")

    tokenizer, model = load_model(LLM_MODEL_DIR)
    answer, matches = answer_question(
        question=args.question,
        search_query=args.question,
        engine=engine,
        tokenizer=tokenizer,
        model=model,
        top_n=TOP_K,
    )

    print(answer)
    print("\nSources:")
    for chunk, score in matches:
        print(f"- {chunk.source} ({score:.3f})")


if __name__ == "__main__":
    main()
