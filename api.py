from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from src.llms.llm import get_response
from src.prompt.prompt import get_prompt
from src.retriever.main import Retriever

app = FastAPI()

model_path = '/home/link/spaces/chunking/LinhCSE/models/retriever'
database_path = '/home/link/spaces/chunking/LinhCSE/data/concat'
retriever = Retriever(
    model_path=model_path, database_path=database_path,
)


def e2e_response(text: str):
    document = retriever.retrieval(text, max_length_output=4096)
    # print(document)
    prompt = get_prompt(query=text, document=document)
    response = get_response(prompt, model_name='gpt4')
    return response


def retrieval_response(text: str):
    document = retriever.retrieval(text, max_length_output=4096)
    return document


def prompt_response(text: str):
    document = retriever.retrieval(text, max_length_output=4096)
    prompt = get_prompt(query=text, document=document)
    return prompt


class Query(BaseModel):
    text: str


@app.get('/')
async def root():
    return {'message': 'Hello World'}


@app.post('/retrieval_docs')
async def get_docs(item: Query):
    return retrieval_response(item.text)


@app.post('/prompting')
async def create_prompt(item: Query):
    return prompt_response(item.text)


@app.post('/e2e_response')
def create_item(item: Query):
    return e2e_response(item.text)
    # return "hello"
