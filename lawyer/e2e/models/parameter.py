from pydantic import BaseModel, Field


class Parameter(BaseModel):
    name: str
    retriever_model_path_or_name: str
    generative_model_path_or_name: str
    database_path: str
    retrieval_max_length: int
    isSeletected: bool = Field(default=False)
