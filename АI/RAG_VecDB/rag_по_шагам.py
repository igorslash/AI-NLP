# ========== ШАБЛОН RAG (как паттерн) ==========
# Аналог LightningModule в PyTorch

class RAGBase:
    """
    4 обязательных этапа, как крестный отец:
    1. LOAD   - загрузить документы
    2. SPLIT  - разбить на куски
    3. INDEX  - превратить в эмбеддинги + сохранить
    4. QUERY  - найти похожее + спросить LLM
    """

    def __init__(self):
        self.documents = None  # как dataset
        self.index = None  # как model state dict
        self.retriever = None  # как forward()
        self.llm = None  # как loss function

    # ЭТАП 1: LOAD (как DataLoader)
    def load_documents(self, path):
        # читаем PDF, DOCX, TXT...
        pass

    # ЭТАП 2: SPLIT (как transform/preprocess)
    def split_documents(self, chunk_size=500):
        # режем на куски
        pass

    # ЭТАП 3: INDEX (как model.fit())
    def build_index(self):
        # считаем эмбеддинги
        # строим векторную БД
        pass

    # ЭТАП 4: QUERY (как model.predict())
    def query(self, question):
        # 1. найти похожие куски
        # 2. отдать в LLM
        # 3. вернуть ответ
        pass


# ========== РЕАЛИЗАЦИЯ ШАБЛОНА (всегда одинаково) ==========
class ConcreteRAG(RAGBase):
    def load_documents(self, path):
        from llama_index.core import SimpleDirectoryReader
        self.documents = SimpleDirectoryReader(path).load_data()

    def split_documents(self, chunk_size=500):
        from llama_index.core.node_parser import SimpleNodeParser
        parser = SimpleNodeParser(chunk_size=chunk_size)
        self.nodes = parser.get_nodes_from_documents(self.documents)

    def build_index(self):
        from llama_index.core import VectorStoreIndex
        self.index = VectorStoreIndex(self.nodes)
        self.retriever = self.index.as_retriever(similarity_top_k=3)
        self.query_engine = self.index.as_query_engine()

    def query(self, question):
        return self.query_engine.query(question)


# ИСПОЛЬЗОВАНИЕ (всегда одинаково)
rag = ConcreteRAG()
rag.load_documents("./docs")
rag.split_documents(500)
rag.build_index()
answer = rag.query("Сколько лет?")