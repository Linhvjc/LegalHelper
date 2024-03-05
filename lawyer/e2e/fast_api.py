from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from api import API

app = FastAPI()
api = API(
    retriever_path='linhphanff/phobert-cse-legal-v1',
    database_path='data/database',
    retrieval_max_length=1024,
)


class Query(BaseModel):
    text: str


@app.get('/')
async def root():
    return {'message': 'Hello World'}


@app.post('/retrieval_docs')
async def get_docs(item: Query):
    return api.retrieval_response(item.text)


@app.post('/prompting')
async def create_prompt(item: Query):
    return api.prompt_response(item.text)


@app.post('/e2e_response')
def e2e(item: Query):
    return api.e2e_response(item.text)

@app.post('/llm_test')
def llm_test(item: Query):
    return api.llm_testing(item.text)