from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from api import API

"""
LLMs model name:

# gpt-3.5
'gpt-3.5-turbo'          : gpt_35_turbo,
'gpt-3.5-turbo-0613'     : gpt_35_turbo_0613,
'gpt-3.5-turbo-16k'      : gpt_35_turbo_16k,
'gpt-3.5-turbo-16k-0613' : gpt_35_turbo_16k_0613,

'gpt-3.5-long': gpt_35_long,

# gpt-4
'gpt-4'          : gpt_4,
'gpt-4-0613'     : gpt_4_0613,
'gpt-4-32k'      : gpt_4_32k,
'gpt-4-32k-0613' : gpt_4_32k_0613,
'gpt-4-turbo'    : gpt_4_turbo,

# Llama 2
'llama2-7b' : llama2_7b,
'llama2-13b': llama2_13b,
'llama2-70b': llama2_70b,
'codellama-34b-instruct': codellama_34b_instruct,
'codellama-70b-instruct': codellama_70b_instruct,

'mixtral-8x7b': mixtral_8x7b,
'mistral-7b': mistral_7b,
'dolphin-mixtral-8x7b': dolphin_mixtral_8x7b,
'lzlv-70b': lzlv_70b,
'airoboros-70b': airoboros_70b,
'airoboros-l2-70b': airoboros_l2_70b,
'openchat_3.5': openchat_35,
'gemini': gemini,
'gemini-pro': gemini_pro,
'claude-v2': claude_v2,
'claude-3-opus': claude_3_opus,
'claude-3-sonnet': claude_3_sonnet,
'pi': pi
"""

app = FastAPI()
api = API(
    retriever_path='linhphanff/phobert-cse-legal-v1',
    database_path='database',
    retrieval_max_length=2048,
    llm_model_name='gpt35'
)

class History(BaseModel):
    arr: str

class Query(BaseModel):
    text: str


@app.get('/')
async def root():
    return "Hello world"


@app.post('/retrieval_docs')
async def get_docs(item: Query):
    return api.retrieval_response(item.text)


@app.post('/prompting')
async def create_prompt(item: Query):
    return api.prompt_response(item.text)


@app.post('/e2e_response')
def e2e(history: History, query: Query):
    return api.e2e_response(history.arr, query.text)

@app.post('/llm_test')
def llm_test(item: Query):
    return api.llm_testing(item.text)