from __future__ import annotations


def get_prompt(query: str, document: str):
    prompt = f"""Answer the question as truthfully and concisely as possible using the provided context, and if the answer is not contained within the text below, say "Tôi không biết". The output must be in VietNamese
    Context: ```{document}```
    Question: ```{query}```
    Answer:"""

    return prompt
