from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from src.llms.llm import get_response
from src.prompt.prompt import get_prompt
from src.retriever.main import Retriever

model_path = '/home/link/spaces/LinhCSE/models/retriever'
corpus_path = '/home/link/spaces/LinhCSE/data/full/corpus.json'
embedding_path = '/home/link/spaces/LinhCSE/data/full/embeddings_corpus.npy'
retriever = Retriever(
    model_path=model_path, corpus_path=corpus_path, embedding_path=embedding_path,
)


def e2e_response(text: str):
    document = retriever.retrieval(text)
    prompt = get_prompt(query=text, document=document)
    response = get_response(prompt)
    return response


# class Query(BaseModel):
#     text: str


# @app.get('/')
# async def root():
#     return {'message': 'Hello World'}


# @app.post('/get_response')
# async def create_item(item: Query):
#     return e2e_response(item.text)
#     # return "hello"

if __name__ == '__main__':
    while True:
        text = input('You: ')
        response = e2e_response(text=text)
        print('Response: ', response)
