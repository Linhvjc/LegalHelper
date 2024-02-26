from __future__ import annotations

from src.modules.llms import LLMs
from src.modules.prompt import Prompt
from src.modules.retriever import Retriever


class API:
    def __init__(
        self,
        retriever_path,
        database_path,
        retrieval_max_length,
    ) -> None:
        self.retriever = Retriever(
            model_path=retriever_path,
            database_path=database_path,
            retrieval_max_length=retrieval_max_length,
        )
        self.prompt = Prompt()
        self.llms = LLMs(model_name='gpt4')

    def e2e_response(self, text: str):
        document = self.retriever.retrieval(text)
        prompt = self.prompt.get_prompt(query=text, document=document)
        response = self.llms.get_response(prompt)
        self.prompt.append_history(user=text, bot=response)
        return response

    def retrieval_response(self, text: str):
        document = self.retriever.retrieval(text)
        return document

    def prompt_response(self, text: str):
        document = self.retriever.retrieval(text)
        prompt = self.prompt.get_prompt(query=text, document=document)
        return prompt
