from fastapi import APIRouter
from models.prompt import Prompt
from config.database import PROMPT_COLLECTION
from schema.schemas import list_serial, individual_serial
from bson import ObjectId

router = APIRouter(prefix="/prompt", tags=["Prompt"])

@router.get("/get_all")
async def get_prompts():
    try:
        prompts = list_serial(PROMPT_COLLECTION.find())
        return prompts
    except Exception as e:
        return f"Error: {e}"

@router.post("/add")
async def post_prompt(prompt: Prompt):
    try:
        PROMPT_COLLECTION.insert_one(dict(prompt))
        return "Add new prompt successfully"
    except Exception as e:
        return f"Error: {e}"

@router.put("/edit/{id}")
async def put_prompt(id: str, prompt: Prompt):
    try:
        PROMPT_COLLECTION.find_one_and_update(
            {"_id": ObjectId(id)}, {"$set": dict(prompt)})
        return f"Edit prompt with id {id} successfully"
    except Exception as e:
        return f"Error: {e}"

@router.delete("/delete/{id}")
async def delete_prompt(id: str):
    try:
        PROMPT_COLLECTION.find_one_and_delete({"_id": ObjectId(id)})
        return f"Update prompt with id {id} successfully"
    except Exception as e:
        return f"Error: {e}"


@router.get("/get_selected")
async def get_selected_prompts():
    try:
        prompts = list_serial(PROMPT_COLLECTION.find({"isSelected": True}))
        return prompts[0]
    except Exception as e:
        return f"Error: {e}"
