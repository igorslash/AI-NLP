from llama_index.core import VectorStoreIndex, Document, Settings, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.vector_stores.qdrant import QdrantVectorStore
import qdrant_client

from app.config import EMBED_MODEL_NAME, LLM_MODEL_NAME, COLLECTION_NAME, MAX_NEW_TOKENS, CONTEXT_WINDOW


class RAGSystem:
    def __init__(self, documents: list[str]):
        # Настройка модели эмбеддингов
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=EMBED_MODEL_NAME
        )
        
        # Настройка LLM
        Settings.llm = HuggingFaceLLM(
            model_name=LLM_MODEL_NAME,
            tokenizer_name=LLM_MODEL_NAME,
            context_window=CONTEXT_WINDOW,
            max_new_tokens=MAX_NEW_TOKENS,
        )
        
        # Подключение Qdrant
        client = qdrant_client.QdrantClient(":memory:")
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Создание документов
        docs = [Document(text=doc) for doc in documents]
        
        # Построение индекса
        self.index = VectorStoreIndex.from_documents(
            docs,
            storage_context=storage_context
        )
        
        # Создание query engine
        self.query_engine = self.index.as_query_engine()

    def query(self, text: str) -> str:
        # Валидация
        if not isinstance(text, str):
            return "Некорректный запрос"
        if not text.strip():
            return "Некорректный запрос"
        
        # Запрос к RAG
        return str(self.query_engine.query(text))

    def add_document(self, text: str) -> None:
        doc = Document(text=text)
        self.index.insert(doc)