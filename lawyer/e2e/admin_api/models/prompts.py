from pydantic import BaseModel

class Prompt(BaseModel):
    name: str
    description: str
    content: str