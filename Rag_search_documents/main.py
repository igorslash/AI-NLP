from fastapi import FastAPI

from app.rag import RAGSystem
from app.models import QueryRequest, QueryResponse

app = FastAPI(title="RAG System")

# Создаём RAG при старте приложения
rag = RAGSystem([
    "Python это язык программирования",
    "Как приготовить борщ",
    "Программирование на Python",
])

@app.post("/query")
def query(request: QueryRequest) -> QueryResponse:
    answer = rag.query(request.text)
    return QueryResponse(answer=answer, status="ok")

@app.get("/health")
def health():
    return {"status": "healthy"}