from __future__ import annotations

from services.modules.llms import LLMs
from services.modules.prompt import Prompt
from services.modules.retriever import Retriever


class ChatController:
    def __init__(
        self,
        retriever_path,
        database_path,
        retrieval_max_length,
        llm_model_name,
        history_max_length = 512
    ) -> None:
        self.retriever = Retriever(
            model_path=retriever_path,
            database_path=database_path,
            retrieval_max_length=retrieval_max_length,
        )
        self.prompt = Prompt()
        self.llms = LLMs(model_name=llm_model_name)
        self.history_max_length = history_max_length

    def e2e_response(self, history: str, text: str):
        try:
            history = eval(history)
            current_history = ''
            for item in history[::-1]:
                if item['role'] == 'assistant':
                    content, relevant = item['content'].split("|||")
                    relevant = " ".join(relevant.split()[:64])
                    current_history = f"{item['role']}: {content}, {relevant}\n" + current_history
                else:
                    current_history = f"{item['role']}: {item['content']}\n" + current_history
                
                if len(current_history.split()) > self.history_max_length:
                    break
        except:
            current_history = history

        document = self.retriever.retrieval(text)
        prompt = self.prompt.get_prompt(query=text, document=document, history=current_history)
        response = self.llms.get_response(prompt)
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
