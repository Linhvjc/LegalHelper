from pydantic import BaseModel, Field


class Prompt(BaseModel):
    name: str
    description: str
    content: str
    isSelected: bool = Field(default=False)
