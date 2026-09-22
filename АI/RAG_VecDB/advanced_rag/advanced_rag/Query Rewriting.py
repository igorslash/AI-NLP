from llama_index.core import Settings
from llama_index.llms.huggingface import HuggingFaceLLM

Settings.llm = HuggingFaceLLM(
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
    tokenizer_name="Qwen/Qwen2.5-0.5B-Instruct",
    context_window=2048,
    max_new_tokens=100,
)

def rewrite_query(query: str) -> str:
    """Переписывает запрос для лучшего поиска."""
    prompt = f"""
    Переформулируй запрос для поиска по базе знаний.
    Сделай его более формальным и конкретным.
    
    Запрос: {query}
    
    Переформулированный запрос:
    """
    
    response = Settings.llm.complete(prompt)
    return str(response).strip()