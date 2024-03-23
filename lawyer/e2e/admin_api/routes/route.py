from fastapi import APIRouter
from models.prompts import Prompt
from config.database import collection_name
from schema.schamas import list_serial
from bson import ObjectId

router = APIRouter()

# GET Request Method
@router.get("/")
async def get_prompts():
    prompts = list_serial(collection_name.find())
    return prompts

# POST Request Method
@router.post("/")
async def post_todo(prompt: Prompt):
    collection_name.insert_one(dict(prompt))

# PUT Request Method
@router.put("/{id}")
async def put_prompt(id: str, prompt: Prompt):
    collection_name.find_one_and_update({"_id": ObjectId(id)}, {"$set": dict(prompt)})

# Delete Request Method
@router.delete("/{id}")
async def delete_prompt(id: str):
    collection_name.find_one_and_delete({"_id": ObjectId(id)})