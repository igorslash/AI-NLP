from langchain_community.tools.tavily_search import TavilySearchResults

tool = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_answer=True
)

# Используется внутри AgentExecutor или create_react_agent
result = tool.invoke("Сравни CrewAI и AutoGen в 2026 году")


# Извлечение чистого контента из списка URL
extract_response = client.extract(
    urls=[
        "https://docs.tavily.com/docs/rest-api/api-reference",
        "https://arxiv.org/abs/2401.12345"
    ]
)

for result in extract_response["results"]:
    print(result["raw_content"])  # Полный чистый текст страницы