from __future__ import annotations


def get_prompt(query: str, document: str):
    prompt = f"""Answer the question as truthfully and concisely as possible using the provided context, and if the answer is not contained within the text below, say "I don't know."
    ---
    The below context is an excerpt from a commercial lease agreement.
    ---
    {document}
    ---
    Question: {query}
    Answer:"""

    return prompt
