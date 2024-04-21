from fastapi import APIRouter
from loguru import logger

from models.message import Message
from models.history import History
from config.database import PARAMETER_COLLECTION
from schema.schemas import list_serial
from schema.chat_controller import ChatController
from services.modules.web_search import web_search

router = APIRouter(prefix="/chatbot", tags=["Chatbot"])

selected_parameter = list_serial(
    PARAMETER_COLLECTION.find({"isSelected": True}))[0]

try:
    controller = ChatController(
        retriever_path=selected_parameter['retriever_model_path_or_name'],
        llm_model_name=selected_parameter['generative_model_path_or_name'],
        database_path=selected_parameter['database_path'],
        retrieval_max_length=int(selected_parameter['retrieval_max_length']),
    )
except Exception as e:
    controller = ChatController(
        retriever_path='linhphanff/phobert-cse-legal-v1',
        llm_model_name='vistral',
        database_path="/home/link/spaces/chunking/LegalHelper/lawyer/e2e/database",
        retrieval_max_length=2048,
    )
    logger.warning("Some parameter doesn't match")
    raise e


@router.post('/retrieval_docs')
async def get_docs(item: Message):
    return await controller.retrieval_response(item.content)


@router.post('/web_search')
async def get_web_search(item: Message):
    return await web_search(item.content)


@router.post('/prompting')
async def create_prompt(item: Message):
    return await controller.prompt_response(item.content)


@router.post('/e2e_response')
async def e2e(history: History, query: Message):
    embedding_query, result = await controller.e2e_response(history.content, query.content)
    # print(embedding_query)
    return embedding_query.tolist()[0], result
    # return embedding_query[0], result


@router.post('/llm_test')
async def llm_test(item: Message):
    return await controller.llm_testing(item.content)


if __name__ == '__main__':
    print('abc')