from pydantic import BaseModel

class QueryRequest(BaseModel):
    text: str

class QueryResponse(BaseModel):
    answer: str
    status: str