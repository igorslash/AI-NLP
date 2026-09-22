# === ЯДРО ===
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
)

# === ПАРСЕРЫ (чанкинг) ===
from llama_index.core.node_parser import SentenceSplitter

# === LLM ===
from llama_index.llms.ollama import Ollama
# from llama_index.llms.openai import OpenAI  # если нужен OpenAI

# === ЭМБЕДДИНГИ ===
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# === ВЕКТОРНЫЕ БАЗЫ ===
from llama_index.vector_stores.chroma import ChromaVectorStore
# from llama_index.vector_stores.qdrant import QdrantVectorStore  # если нужен Qdrant

# === САМА БАЗА ===
import chromadb
# import qdrant_client  # если нужен Qdrant