# 1. Создай структуру тестовых документов
mkdir -p docs/it docs/hr docs/security

echo "# Доступы\nДля получения Jira создай тикет в ServiceDesk (sd.company.com). Доступ к GitHub выдает тимлид." > docs/it/access_guide.md
echo "# ДМС\nПолис ДМС оформляется после испытательного срока. Анкету заполнить в HR-портале." > docs/hr/benefits.md
echo "# Пароли\nЗапрещено использовать личные пароли. Установи 1Password из корпоративного портала." > docs/security/rules.md

# 2. Установка и запуск
pip install -r requirements.txt
uvicorn main:app --reload

# 3. Тестовый запрос (через curl или Swagger на http://127.0.0.1:8000/docs)
curl -X POST "http://localhost:8000/generate-plan" \
     -H "Content-Type: application/json" \
     -d '{"role": "Python Developer", "question": "Что мне нужно сделать в первый день для получения доступов и оформления документов?"}'