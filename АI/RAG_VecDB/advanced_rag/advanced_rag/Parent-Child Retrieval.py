from llama_index.core import Document, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Документ
document = Document(text="""
Python — это язык программирования.
Он используется для веб-разработки, анализа данных и машинного обучения.
Python был создан Гвидо ван Россумом в 1991 году.
С тех пор язык стал одним из самых популярных в мире.
Язык отличается простым синтаксисом и читаемостью кода.
Python поддерживает множество парадигм программирования.
""")

# Сплиттер: маленькие дети
splitter = SentenceSplitter(chunk_size=100, chunk_overlap=20)

# Создаём child-чанки
children = splitter.get_nodes_from_documents([document])

print(f"Родитель: {len(document.text)} символов")
print(f"Детей: {len(children)}")
print()

for i, child in enumerate(children, 1):
    print(f"Ребёнок {i}: {child.text[:80]}...")
    print("---")