from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from src.llms.llm import get_response
from src.prompt.prompt import get_prompt
from src.retriever.main import Retriever

app = FastAPI()

model_path = '/home/link/spaces/LinhCSE/models/retriever'
database_path = '/home/link/spaces/LinhCSE/data/concat'
retriever = Retriever(
    model_path=model_path, database_path=database_path,
)


def e2e_response(text: str):
    document = retriever.retrieval(text)
    print(document)
    prompt = get_prompt(query=text, document=document)
    response = get_response(prompt, model_name='gpt35')
    return response


class Query(BaseModel):
    text: str


@app.get('/')
async def root():
    return {'message': 'Hello World'}


@app.post('/get_response')
def create_item(item: Query):
    return e2e_response(item.text)
    # return "hello"
