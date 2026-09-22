# ============================================
# МОЙ ШАБЛОН RAG (работает всегда)
# ============================================

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

# 1. Загрузить документы (один раз)
documents = SimpleDirectoryReader("./docs").load_data()

# 2. Создать индекс (один раз, долго)
index = VectorStoreIndex.from_documents(documents)

# 3. Сохранить/загрузить индекс (чтобы не пересоздавать)
index.storage_context.persist("./storage")
# при следующем запуске:
# from llama_index.core import load_index_from_storage
# index = load_index_from_storage(StorageContext.from_defaults(persist_dir="./storage"))

# 4. Спросить (быстро)
query_engine = index.as_query_engine()
answer = query_engine.query("Ваш вопрос")

# 5. Профит!