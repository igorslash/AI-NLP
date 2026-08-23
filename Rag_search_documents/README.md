# RAG System

RAG-система на LlamaIndex + Qdrant с API на FastAPI.

## Возможности
- Векторный поиск по документам
- Добавление документов на лету
- Валидация запросов
- Health-check

## Стек
- LlamaIndex
- Qdrant
- FastAPI
- Hugging Face (all-MiniLM-L6-v2, Qwen2.5-0.5B)

## Запуск

### Локально
pip install -r requirements.txt
uvicorn app.main:app --reload

### Docker
docker build -t rag-system .
docker run -p 8000:8000 rag-system

## API

### POST /query
Тело: {"text": "Что такое Python?"}
Ответ: {"answer": "...", "status": "ok"}

### GET /health
Ответ: {"status": "healthy"}

## Тесты
pytest test_api.py