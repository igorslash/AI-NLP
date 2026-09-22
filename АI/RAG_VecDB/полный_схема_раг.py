# main.py - ЭТО ВСЕ, ЧТО ТЕБЕ НУЖНО ДЛЯ 90% ЗАДАЧ

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings

# 1. НАСТРОЙКА (один раз)
Settings.llm = Ollama(model="llama3.2:3b", temperature=0.1)

# 2. ЗАГРУЗИТЬ PDF
documents = SimpleDirectoryReader("./docs").load_data()

# 3. СОЗДАТЬ ИНДЕКС (делает чанкинг + эмбеддинги)
index = VectorStoreIndex.from_documents(documents)

# 4. ЗАДАТЬ ВОПРОС
response = index.as_query_engine().query("Какой опыт работы?")

# 5. ПОЛУЧИТЬ ОТВЕТ
print(response)