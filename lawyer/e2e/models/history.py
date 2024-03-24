from pydantic import BaseModel


class History(BaseModel):
    content: str
