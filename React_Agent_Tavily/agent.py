import os
from llama_index.core.agent import ReActAgent
from llama_index.tools.tavily_research import TavilyResearchToolSpec
from llama_index.llms.openai import OpenAI

# === 1. Инициализация ===
llm = OpenAI(model="gpt-4o", temperature=0.3)

tavily_tools = TavilyResearchToolSpec(
    api_key=os.environ["TAVILY_API_KEY"]
).to_tool_list()

# === 2. Создание агента ===
agent = ReActAgent.from_tools(
    tools=tavily_tools,
    llm=llm,
    verbose=True,
    max_iterations=5,  # Защита от бесконечных циклов
    system_prompt="""Ты — эксперт-аналитик. Напиши подробный исследовательский 
отчёт на русском языке.

Структура:
1. Введение
2. Основные выводы (3-5 пунктов)
3. Детальный анализ
4. Противоречия и пробелы в данных
5. Заключение

Ссылайся на источники (URL) прямо в тексте. Минимум 800 слов."""
)

# === 3. Запуск ===
topic = input("Введите тему исследования: ")
response = agent.chat(f"Напиши подробный отчёт на тему: {topic}")

# === 4. Вывод и сохранение ===
print("\n" + "=" * 80)
print(response.response)
print("=" * 80)

filename = f"report_{topic.replace(' ', '_')[:30]}.md"
with open(filename, "w", encoding="utf-8") as f:
    f.write(response.response)
print(f"\n✅ Отчёт сохранён в {filename}")