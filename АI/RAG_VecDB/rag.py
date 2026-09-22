# 0. Установка (запусти один раз в терминале):
# pip install sentence-transformers numpy

from sentence_transformers import SentenceTransformer, util
import numpy as np

# 1. Загружаем модель (скачается автоматически при первом запуске ~400МБ)
# Это наш "переводчик" текста в векторы
model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Модель загружена")

# 2. Наши тестовые фразы
sentences = [
    "Кошка сидит на подоконнике",      # Индекс 0
    "Пёс гоняется за мячом во дворе",  # Индекс 1
    "Срочно нужен отчет по продажам за пятницу"  # Индекс 2
]

# 3. Превращаем текст в векторы (эмбеддинги)
# Каждый текст → массив из 384 чисел (координат в пространстве смыслов)
embeddings = model.encode(sentences, convert_to_tensor=True)
print(f"✅ Векторы получены. Размер одного вектора: {embeddings[0].shape}")

# 4. Считаем косинусное сходство (близость смыслов)
# Результат: матрица 3x3, где [i][j] — сходство между фразой i и фразой j
cosine_scores = util.cos_sim(embeddings, embeddings)

# 5. Выводим результаты в понятном виде
print("\n🔍 Результаты поиска смысла:")
print(f"{'Фраза 1':<40} | {'Фраза 2':<40} | Сходство")
print("-" * 90)

for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):  # Проходим только по уникальным парам
        score = cosine_scores[i][j].item()  # Извлекаем число из тензора
        print(f"{sentences[i]:<40} | {sentences[j]:<40} | {score:.3f}")