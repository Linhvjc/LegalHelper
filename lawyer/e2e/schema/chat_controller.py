from __future__ import annotations

from services.modules.llms import LLMs
from services.modules.prompt import Prompt
from services.modules.retriever import Retriever
from services.modules.web_search import web_search


class ChatController:
    def __init__(
        self,
        retriever_path,
        database_path,
        retrieval_max_length,
        llm_model_name,
        history_max_length=1024
    ) -> None:
        self.retriever = Retriever(
            model_path=retriever_path,
            database_path=database_path,
            retrieval_max_length=retrieval_max_length,
        )
        self.prompt = Prompt()
        self.llms = LLMs(model_name=llm_model_name)
        self.history_max_length = history_max_length

    async def e2e_response(self, history: str, text: str):
        try:
            history = eval(history)
            current_history = ''
            for item in history[::-1]:
                if item['role'] == 'assistant':
                    content, relevant = item['content'].split("|||")
                    relevant = " ".join(relevant.split()[:64])
                    current_history = f"{item['role']}: {content}, {relevant}\n" + \
                        current_history
                else:
                    current_history = f"{item['role']}: {item['content']}\n" + \
                        current_history

                if len(current_history.split()) > self.history_max_length:
                    break
        except:
            current_history = history

        document = await self.retriever.retrieval(text)
        document_search = await web_search(text)
        prompt = await self.prompt.get_prompt_vi(query=text,
                                           document=document,
                                           history=current_history,
                                           document_search=document_search)
        print("Len: ",len(prompt.split()))
        response = await self.llms.get_response(prompt)
        return f"{response}|||Relevant doc: {document}"

    async def retrieval_response(self, text: str):
        document = await self.retriever.retrieval(text)
        return document

    def prompt_response(self, text: str):
        document = self.retriever.retrieval(text)
        prompt = self.prompt.get_prompt_vi(
            query=text, document=document, history='', document_search='')
        return prompt

    def llm_testing(self, text: str):
        return self.llms.get_response(text)


if __name__ == '__main__':
    print('chat controller')
