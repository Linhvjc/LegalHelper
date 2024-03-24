from pydantic import BaseModel, Field


class Prompt(BaseModel):
    name: str
    description: str
    content: str
    isSeletected: bool = Field(default=False)
