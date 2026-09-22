import os
from llama_index.core.agent import ReActAgent
from llama_index.tools.tavily_research import TavilyResearchToolSpec
from llama_index.llms.openai import OpenAI

# === Инициализация ===
llm = OpenAI(model="gpt-4o", temperature=0.3)

# Tavily Research Tool — это ГОТОВЫЙ инструмент, который УЖЕ делает
# search(advanced) + extract(top URLs) + синтез внутри себя
tavily_tools = TavilyResearchToolSpec(
    api_key=os.environ["TAVILY_API_KEY"]
).to_tool_list()

# === Создание агента ===
agent = ReActAgent.from_tools(
    tools=tavily_tools,
    llm=llm,
    verbose=True,          # Показывает ход мыслей агента
    system_prompt="""Ты — эксперт-аналитик. Твоя задача — писать ПОДРОБНЫЕ 
исследовательские отчёты на русском языке.

Правила:
1. Всегда используй инструмент поиска для получения актуальной информации
2. Структура отчёта: Введение → Основные выводы → Детальный анализ → 
   Противоречия/пробелы → Заключение
3. Ссылайся на источники (URL) прямо в тексте
4. Минимум 1500 слов
5. Если информация противоречива — явно укажи это"""
)

# === Запуск ===
topic = input("Введите тему исследования: ")
response = agent.chat(f"Напиши подробный исследовательский отчёт на тему: {topic}")

print("\n" + "=" * 80)
print(response.response)
print("=" * 80)