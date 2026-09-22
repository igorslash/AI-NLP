from llama_index.core import Settings
from llama_index.llms.huggingface import HuggingFaceLLM

Settings.llm = HuggingFaceLLM(
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
    tokenizer_name="Qwen/Qwen2.5-0.5B-Instruct",
    context_window=2048,
    max_new_tokens=200,
)

def compress_context(query: str, context: str) -> str:
    """Сжимает контекст, оставляя только релевантное."""
    prompt = f"""
    Вопрос: {query}
    
    Контекст:
    {context}
    
    Извлеки из контекста только ту информацию, которая отвечает на вопрос.
    Убери всё лишнее.
    
    Сжатый контекст:
    """
    
    response = Settings.llm.complete(prompt)
    return str(response).strip()