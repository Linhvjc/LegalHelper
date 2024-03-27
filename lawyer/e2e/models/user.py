from pydantic import BaseModel, Field


class User(BaseModel):
    name: str
    email: str
