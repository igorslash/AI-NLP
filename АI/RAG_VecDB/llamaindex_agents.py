import asyncio
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import FunctionTool
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.ollama import Ollama

# ========== 1. RAG КАК ИНСТРУМЕНТ (добавили 5 строк) ==========
def search_in_pdf(question: str) -> str:
    """Ищет ответ в PDF документах компании"""
    documents = SimpleDirectoryReader("./docs").load_data()  # загрузить PDF
    index = VectorStoreIndex.from_documents(documents)      # создать индекс
    query_engine = index.as_query_engine()                  # создать поиск
    return str(query_engine.query(question))                # вернуть ответ

# ========== 2. ТВОЙ ОРИГИНАЛЬНЫЙ КОД (почти без изменений) ==========
def multiply(a: float, b: float) -> float:
    """Useful for multiplying two numbers."""
    return a * b

agent = FunctionAgent(
    tools=[
        FunctionTool.from_defaults(search_in_pdf),  # ← RAG как инструмент
        FunctionTool.from_defaults(multiply),        # ← математика
    ],
    llm=Ollama(
        model="llama3.1",
        request_timeout=360.0,
        context_window=8000,
    ),
    system_prompt="""You are a helpful assistant. 
    Use search_in_pdf to answer questions about company documents.
    Use multiply for math operations.
    """,
)

async def main():
    # Агент сам решит, какой инструмент использовать!
    response = await agent.run("What is 1234 * 4567?")        # пойдет в multiply
    print(response)
    
    response = await agent.run("What skills are in the resume?") # пойдет в search_in_pdf
    print(response)

if __name__ == "__main__":
    asyncio.run(main())