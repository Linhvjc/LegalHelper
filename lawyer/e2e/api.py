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
        llm_model_name
    ) -> None:
        self.retriever = Retriever(
            model_path=retriever_path,
            database_path=database_path,
            retrieval_max_length=retrieval_max_length,
        )
        self.prompt = Prompt()
        self.llms = LLMs(model_name=llm_model_name)

    def e2e_response(self, history: str, text: str):
        try:
            history = eval(history)
            current_history = ''
            for item in history:
                if item['role'] == 'assistant':
                    content, relevant = item['content'].split("|||")
                    current_history += f"{item['role']}: {content}, {relevant}\n"
                else:
                    current_history += f"{item['role']}: {item['content']}\n"
        except:
            current_history = history

        document = self.retriever.retrieval(text)
        prompt = self.prompt.get_prompt(query=text, document=document, history=current_history)
        response = self.llms.get_response(prompt)
        # self.prompt.append_history(user=text, bot=response)
        return f"{response}|||Relevant doc: {document}"

    def retrieval_response(self, text: str):
        document = self.retriever.retrieval(text)
        return document

    def prompt_response(self, text: str):
        document = self.retriever.retrieval(text)
        prompt = self.prompt.get_prompt(query=text, document=document, history='')
        return prompt
    
    def llm_testing (self, text: str):
        return self.llms.get_response(text)
