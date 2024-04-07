from __future__ import annotations
import asyncio
import time

from services.modules.llms import LLMs
from services.modules.prompt import Prompt
from services.modules.retriever import Retriever
from services.modules.web_search import web_search
from services.utils.utils import get_short_term_memory
from services.utils.utils import get_long_term_memory


class ChatController:
    def __init__(
        self,
        retriever_path,
        database_path,
        retrieval_max_length,
        llm_model_name,
        history_max_length=256
    ) -> None:
        self.retriever = Retriever(
            model_path=retriever_path,
            database_path=database_path,
            retrieval_max_length=retrieval_max_length,
        )
        self.prompt = Prompt()
        self.llms = LLMs(model_name=llm_model_name)
        self.history_max_length = history_max_length
        self.retrieval_system_prompt = self.prompt.get_retrieval_system_prompt()

    async def e2e_response(self, history: str, text: str):
        start_time = time.time()
        
        (history, short_term_memory), (embedding_query, document), document_search = await asyncio.gather(
            get_short_term_memory(history),
            self.retriever.retrieval_asym(text),
            web_search(text)
        )
        long_term_memory = await get_long_term_memory(embedding_query=embedding_query,
                                                    history=history)
        prompt = await self.prompt.get_prompt_vi(query=text,
                                                document=document,
                                                short_term_history=short_term_memory,
                                                long_term_history=long_term_memory,
                                                document_search=document_search)
        print("Len: ",len(prompt.split()))
        response = await self.llms.get_response(message=prompt, system = self.retrieval_system_prompt)
        
        end_time = time.time()
        response_time = end_time - start_time
        print("Response time:", response_time, "seconds")
        
        return embedding_query, f"{response}|||Relevant doc: {document}"

    async def retrieval_response(self, text: str):
        embedding_query, document = await self.retriever.retrieval_asym(text)
        return embedding_query, document

    def prompt_response(self, text: str):
        embedding_query, document = self.retriever.retrieval_asym(text)
        prompt = self.prompt.get_prompt_vi(
            query=text, 
            document=document, 
            short_term_history='',
            long_term_history='',
            document_search='')
        return prompt

    def llm_testing(self, text: str):
        return self.llms.get_response(text)


if __name__ == '__main__':
    print('chat controller')
