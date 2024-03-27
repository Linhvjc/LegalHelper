from fastapi import APIRouter
from models.parameter import Parameter
from config.database import PARAMETER_COLLECTION
from schema.schemas import list_serial
from bson import ObjectId

router = APIRouter(prefix="/parameter", tags=["Parameter"])


@router.get("/get_all")
async def get_parameters():
    try:
        parametes = list_serial(PARAMETER_COLLECTION.find())
        return parametes
    except Exception as e:
        return f"Error: {e}"


@router.post("/add")
async def post_parameter(parameter: Parameter):
    try:
        PARAMETER_COLLECTION.insert_one(dict(parameter))
    except Exception as e:
        return f"Error: {e}"
    


@router.put("/edit/{id}")
async def put_parameter(id: str, parameter: Parameter):
    try:
        PARAMETER_COLLECTION.find_one_and_update(
            {"_id": ObjectId(id)}, {"$set": dict(parameter)})
    except Exception as e:
        return f"Error: {e}"


@router.delete("/delete/{id}")
async def delete_parameter(id: str):
    try:
        PARAMETER_COLLECTION.find_one_and_delete({"_id": ObjectId(id)})
    except Exception as e:
        return f"Error: {e}"


@router.get("/get_selected")
async def get_selected_parameters():
    try:
        parameters = list_serial(
            PARAMETER_COLLECTION.find({"isSelected": True}))
        return parameters[0]
    except Exception as e:
        return f"Error: {e}"


@router.post("/set_selected/{id}")
async def set_selected_parameter(id: str):
    try:
        PARAMETER_COLLECTION.update_many({}, {"$set": {"isSelected": False}})

        PARAMETER_COLLECTION.find_one_and_update(
            {"_id": ObjectId(id)}, {"$set": {"isSelected": True}})

        return {"message": "Selected parameter updated successfully."}
    except Exception as e:
        return f"Error: {e}"
