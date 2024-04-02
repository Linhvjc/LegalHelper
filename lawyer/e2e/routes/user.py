from fastapi import APIRouter
from models.user import User
from config.database import USER_COLLECTION
from schema.schemas import list_serial
from bson import ObjectId

router = APIRouter(prefix="/user", tags=["User"])


@router.get("/get_all")
async def get_users():
    try:
        users = list_serial(USER_COLLECTION.find())
        return users
    except Exception as e:
        return f"Error: {e}"


@router.post("/add")
async def post_user(user: User):
    try:
        USER_COLLECTION.insert_one(dict(user))
    except Exception as e:
        return f"Error: {e}"
    


@router.put("/edit/{id}")
async def put_user(id: str, user: User):
    try:
        USER_COLLECTION.find_one_and_update(
            {"_id": ObjectId(id)}, {"$set": dict(user)})
    except Exception as e:
        return f"Error: {e}"


@router.delete("/delete/{id}")
async def delete_user(id: str):
    try:
        USER_COLLECTION.find_one_and_delete({"_id": ObjectId(id)})
    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    print('abc')
