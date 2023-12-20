from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from src.retriever.main import Retriever

app = FastAPI()

model_path = '/home/link/spaces/LinhCSE/models/retriever'
corpus_path = '/home/link/spaces/LinhCSE/data/full/corpus.json'
embedding_path = '/home/link/spaces/LinhCSE/data/full/embeddings_corpus.npy'
retriever = Retriever(
    model_path=model_path, corpus_path=corpus_path, embedding_path=embedding_path,
)


class Query(BaseModel):
    text: str


@app.get('/')
async def root():
    return {'message': 'Hello World'}


@app.post('/get_response')
async def create_item(item: Query):
    return retriever.retrieval(item.text)
