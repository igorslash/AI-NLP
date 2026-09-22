import re
import torch
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================
# 1. ОЧИСТКА ТЕКСТА
# ============================================

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^а-яё\s]", "", text)  # только русские буквы и пробелы
    text = re.sub(r"\s+", " ", text)  # схлопываем пробелы
    return text.strip()


# ============================================
# 2. ЗАГРУЗКА И ЧАНКИНГ
# ============================================

raw_text = """
Глава 1. Введение в RAG. 
RAG (Retrieval-Augmented Generation) — это метод, который улучшает ответы LLM 
за счёт поиска релевантной информации в базе знаний.

--- Стр. 1 --- 

Механизм RAG состоит из двух этапов:
1. Поиск (Retrieval) — находит документы, похожие на вопрос пользователя.
2. Генерация (Generation) — LLM формирует ответ на основе найденных документов.

--- Стр. 2 --- 

Этот подход снижает галлюцинации модели и позволяет использовать актуальные данные.
"""

cleaned_text = clean_text(raw_text)

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_text(cleaned_text)

print(f"📄 Очищенный текст:\n{cleaned_text}\n")
print(f"📦 Получено чанков: {len(chunks)}")

# ============================================
# 3. ЭМБЕДДИНГИ И ВЕКТОРНАЯ БД
# ============================================

# Мультиязычная модель для русского
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
chunk_embeddings = model.encode(chunks, convert_to_tensor=True)

db = FAISS.from_texts(chunks, model)


# ============================================
# 4. LLM ДЛЯ ГЕНЕРАЦИИ (GPT-2 на русском понимает плохо, но для примера)
# ============================================

class SimpleLLM:
    def __init__(self, model_name="gpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    def generate(self, prompt, max_length=200):
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                inputs,
                max_length=max_length,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )

        return self.tokenizer.decode(output[0], skip_special_tokens=True)


# ============================================
# 5. RAG ПАЙПЛАЙН
# ============================================

def rag_answer(question, db, llm, k=3):
    # 1. Поиск релевантных чанков
    docs = db.similarity_search(question, k=k)

    if not docs:
        return "Не нашёл relevant документов."

    # 2. Формируем контекст
    context = "\n\n---\n\n".join(docs)

    # 3. Промпт для LLM (на русском, но GPT-2 на русском плохо, лучше бы взять ruGPT)
    prompt = f"""Ты — полезный ассистент. Ответь на вопрос, используя только этот контекст. Если ответа нет в контексте, честно скажи "Я не знаю".

Контекст:
{context}

Вопрос: {question}

Ответ:"""

    # 4. Генерация
    answer = llm.generate(prompt, max_length=250)

    # 5. Отрезаем сам вопрос, если он попал в ответ (для GPT-2 это часто)
    if question.lower() in answer.lower():
        answer = answer.split(question)[-1]

    return {
        "question": question,
        "answer": answer.strip(),
        "retrieved_docs": docs
    }


# ============================================
# 6. ЗАПУСК
# ============================================

llm = SimpleLLM()
question = "Что такое RAG?"

result = rag_answer(question, db, llm)

print("\n" + "=" * 50)
print(f"❓ Вопрос: {result['question']}")
print(f"📚 Найдено документов: {len(result['retrieved_docs'])}")
print("-" * 30)
print(f"🤖 Ответ: {result['answer']}")
print("=" * 50)

# Можно задать ещё вопрос
question2 = "Из каких этапов состоит механизм RAG?"
result2 = rag_answer(question2, db, llm)

print(f"\n❓ Вопрос: {result2['question']}")
print(f"🤖 Ответ: {result2['answer']}")