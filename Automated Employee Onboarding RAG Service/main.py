from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from onboarding_engine import OnboardingRAG

app = FastAPI(
    title="Onboarding RAG Assistant",
    description="Микросервис генерации пошаговых планов онбординга на основе регламентов"
)

rag = OnboardingRAG(docs_root="./docs")

class PlanRequest(BaseModel):
    role: str         # Роль сотрудника
    question: str     # Вопрос по онбордингу

@app.post("/generate-plan")
async def get_onboarding_plan(req: PlanRequest):
    if not rag.tools:
        raise HTTPException(status_code=503, detail="База знаний пуста. Добавьте документы в папки.")
    
    result = rag.generate_plan(role=req.role, question=req.question)
    return result

@app.get("/health")
async def health():
    return {"status": "ok", "departments_loaded": len(rag.tools)}