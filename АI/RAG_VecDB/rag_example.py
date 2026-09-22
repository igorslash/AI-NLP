import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import faiss

# ============================================
# 1. RAG ПОИСК (Векторная база + FAISS)
# ============================================

class RAGSearch:
    def __init__(self, documents):
        self.documents = documents
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Создаём эмбеддинги для всех документов
        self.doc_embeddings = self.model.encode(documents, convert_to_numpy=True)
        
        # Нормализуем для косинусного расстояния
        faiss.normalize_L2(self.doc_embeddings)
        
        # Создаём FAISS индекс
        self.index = faiss.IndexFlatL2(self.doc_embeddings.shape[1])
        self.index.add(self.doc_embeddings)
    
    def search(self, query, k=3):
        """Ищет top-k документов по запросу"""
        query_emb = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_emb)
        distances, indices = self.index.search(query_emb, k)
        return [self.documents[i] for i in indices[0]]

# ============================================
# 2. LLM ГЕНЕРАТОР
# ============================================

class GeneratorLLM:
    def __init__(self, model_name="gpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
    
    def generate(self, prompt, max_length=150, temperature=0.7):
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            output = self.model.generate(
                inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

# ============================================
# 3. ПОЛНЫЙ RAG ПАЙПЛАЙН
# ============================================

class RAGPipeline:
    def __init__(self, documents, llm_model_name="gpt2"):
        self.retriever = RAGSearch(documents)
        self.generator = GeneratorLLM(llm_model_name)
    
    def ask(self, question, k=3, max_length=150):
        # 1. Ищем релевантные документы
        relevant_docs = self.retriever.search(question, k=k)
        
        # 2. Формируем промпт
        context = "\n".join(relevant_docs)
        prompt = f"""Ответь на вопрос, используя только этот контекст. Если ответа нет в контексте, скажи "Не знаю".

Контекст:
{context}

Вопрос: {question}

Ответ:"""
        
        # 3. Генерируем ответ
        answer = self.generator.generate(prompt, max_length=max_length)
        
        return {
            "question": question,
            "answer": answer,
            "retrieved_documents": relevant_docs
        }

# ============================================
# 4. ЗАПУСК
# ============================================

if __name__ == "__main__":
    # Наши документы (база знаний)
    documents = [
        "Столица Франции — Париж. Париж также называют городом света.",
        "Эйфелева башня находится в Париже. Она была построена в 1889 году.",
        "Франция славится своими винами и сырами. Бордо — известный винодельческий регион.",
        "Лувр — один из крупнейших музеев мира, расположен в Париже.",
        "Французская кухня включает круассаны, багеты и улиток."
    ]
    
    # Создаём RAG систему
    rag = RAGPipeline(documents)
    
    # Задаём вопросы
    questions = [
        "Какая столица Франции?",
        "Где находится Эйфелева башня?",
        "Чем славится Франция?",
        "Что такое Лувр?"
    ]
    
    for q in questions:
        print("\n" + "="*50)
        result = rag.ask(q)
        print(f"❓ Вопрос: {result['question']}")
        print(f"📚 Найдено документов: {len(result['retrieved_documents'])}")
        print(f"🤖 Ответ: {result['answer']}")
        print("-"*30)