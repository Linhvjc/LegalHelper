from fastapi import APIRouter
from bson import ObjectId
from loguru import logger

from models.parameter import Parameter
from models.message import Message
from models.history import History
from config.database import PARAMETER_COLLECTION
from schema.schemas import list_serial
from schema.chat_controller import ChatController

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
except:
    controller = ChatController(
        retriever_path='linhphanff/phobert-cse-legal-v1',
        llm_model_name='gpt-4-32k',
        database_path="/home/link/spaces/chunking/LegalHelper/lawyer/e2e/database",
        retrieval_max_length=2048,
    )
    logger.warning("Some parameter doesn't match")


@router.post('/retrieval_docs')
async def get_docs(item: Message):
    return controller.retrieval_response(item.content)


@router.post('/prompting')
async def create_prompt(item: Message):
    return controller.prompt_response(item.content)


@router.post('/e2e_response')
def e2e(history: History, query: Message):
    return controller.e2e_response(history.content, query.content)


@router.post('/llm_test')
def llm_test(item: Message):
    return controller.llm_testing(item.content)
