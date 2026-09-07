"""
llm/generator.py - Runs a single inference pass against the loaded model.
"""

from config.models import GENERATION_TEMPERATURE, MAX_NEW_TOKENS
from llm.domain_prompts import SYSTEM_INSTRUCTION, build_domain_instructions
from llm.prompts import build_prompt


def generate(
    question: str,
    context: str,
    tokenizer,
    model,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """Run a single inference pass and return the decoded answer string."""
    if tokenizer is None or model is None:
        raise ValueError("A tokenizer and model are required for generation")

    import torch

    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {
            "role": "user",
            "content": build_prompt(
                question,
                context,
                build_domain_instructions(question),
            ),
        },
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([prompt], return_tensors="pt", padding=True)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            temperature=GENERATION_TEMPERATURE,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# Backwards-compatible alias matching the old rag_engine.py name
generate_answer = generate
