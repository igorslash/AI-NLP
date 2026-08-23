from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_query():
    response = client.post("/query", json={"text": "Что такое Python?"})
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_invalid_query_empty():
    response = client.post("/query", json={"text": ""})
    assert response.json()["answer"] == "Некорректный запрос"

def test_invalid_query_type():
    response = client.post("/query", json={"text": 123})
    assert response.status_code == 422  # Pydantic validation error