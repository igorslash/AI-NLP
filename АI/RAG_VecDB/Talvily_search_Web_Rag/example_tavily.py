from llama_index.core.agent import ReActAgent
from llama_index.tools.tavily_search import TavilySearchToolSpec
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# Локальный индекс
docs = SimpleDirectoryReader("./docs").load_data()
index = VectorStoreIndex.from_documents(docs)
local_tool = index.as_query_engine().as_tool(
    description="Поиск по внутренней документации проекта"
)

# Tavily как инструмент
tavily_tool = TavilySearchToolSpec(
    api_key=os.environ["TAVILY_API_KEY"],
    max_results=5,
    search_depth="advanced"
).to_tool_list()[0]

# Агент сам решает, когда искать в docs, а когда в интернете
agent = ReActAgent.from_tools(
    tools=[local_tool, tavily_tool],
    verbose=True
)

response = agent.chat("Сравни нашу внутреннюю архитектуру с лучшими практиками 2026 года")
print(response)