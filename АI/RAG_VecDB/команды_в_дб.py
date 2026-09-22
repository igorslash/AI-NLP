import chromadb
# Данные сохранятся в папку, а не исчезнут после перезагрузки
client = chromadb.PersistentClient(path="./my_vector_db") 

# Создать новую или загрузить существующую
collection = client.get_or_create_collection(name="manuals")

# Удалить коллекцию (если хочешь начать с чистого листа)
client.delete_collection(name="manuals")

# Добавить в коллекцию
collection.upsert(
    ids=["id1", "id2"],
    documents=["Текст первого чанка", "Текст второго чанка"],
    metadatas=[{"page": 1}, {"page": 2}]
)
# Поиск
results = collection.query(
    query_texts=["Как настроить принтер?"], 
    n_results=3, # Сколько чанков вернуть
    # Можно добавить фильтр по метаданным
    where={"page": {"$gte": 1}} # Например, искать только со страницы 1 и выше
)
## Сколько всего чанков в базе?
print(collection.count())

# Показать первые 5 записей (чтобы проверить чистку и ID)
print(collection.peek(limit=5))


