# Production-Ready AI, NLP & Backend Ecosystem

Welcome to my portfolio repository. This ecosystem demonstrates production-grade integrations of Large Language Models (LLMs), Multi-Agent frameworks, Retrieval-Augmented Generation (RAG), and high-performance text processing pipelines, backed by solid backend architecture.

---

## 🚀 Key Projects & Architecture

### 🦀 High-Performance Core
*   **[fast_text_proc](./fast_text_proc)**: A high-performance text tokenization and normalization library written in **Rust** and exposed to Python via **PyO3**. Features single-pass regex compilation, punctuation removal, lowercasing, and multi-core batch processing using the `Rayon` crate. Designed to eliminate Python bottlenecks in heavy NLP pipelines.

### 🤖 Multi-Agent Systems & AI Integrations
*   **[Insurence_ai_agent](./Insurence_ai_agent)**: An automated insurance claims processing system powered by the **CrewAI** multi-agent framework. Coordinates specialized agents to analyze claims, cross-reference policy guidelines, and generate structured summaries.
*   **[BankSQL-Agent](./BankSQL-Agent)**: A secure Text-to-SQL autonomous agent designed to translate natural language financial queries into executable SQL, interacting directly with bank databases while ensuring schema privacy.
*   **[ApiAssistent](./ApiAssistent)**: An AI-driven API assistant capable of understanding user intent, mapping it to specific API endpoints, and orchestrating complex multi-step backend calls.

### 📚 Knowledge Bases & RAG
*   **[Rag_search_documents](./Rag_search_documents)** & **[rag-knowledge-base](./rag-knowledge-base)**: Production-grade RAG pipelines utilizing **LlamaIndex** and vector databases (ChromaDB/Qdrant). Optimized for chunking, metadata extraction, semantic retrieval, and synthesis over massive unstructured PDF/text documentation.

### 🛡️ Production Infrastructure & Classic Backend
*   **[TodoApiProject](./TodoApiProject)**: A clean architecture backend service built with Python web frameworks, demonstrating robust routing, async programming (`asyncio`), database migrations, and unit testing.
*   **Enterprise Tooling**: Containerized with **Docker** for seamless orchestration and cloud deployment, utilizing strict dependency lockfiles (`Cargo.lock`, `pyproject.toml`).

---

## 🛠️ Tech Stack & Skills

*   **Languages:** Python (Asyncio, FastAPI), Rust (PyO3, Rayon)
*   **AI & LLM Orchestration:** CrewAI, LlamaIndex, LangChain, OpenAI API, Anthropic API
*   **Vector Search & DBs:** ChromaDB, Pinecone, PostgreSQL, SQLAlchemy
*   **DevOps & Infrastructure:** Docker, Docker Compose, CI/CD Baselines, Git Workflow

---

## ⚙️ How to Explore

1. **Rust-Python Binding:** Navigate to `fast_text_proc`, ensure you have the Rust toolchain ready, and build the extension locally using `maturin develop --release`.
2. **AI Agents:** Check out `Insurence_ai_agent` or `BankSQL-Agent` to see how Prompt Engineering, Memory Management, and Tool Calling are orchestrated in real-world scenarios.
3. **Backend & RAG:** Explore `Rag_search_documents` to view the full pipeline from raw data ingestion to context-augmented LLM responses.

---

## 📬 Contact & Collaboration

I specialize in bridging the gap between cutting-edge LLM capabilities and reliable backend engineering. Open for remote contracts, startup roles, and architectural consultations.

*   **GitHub:** [igorslash](https://github.com)
*   **Email:** `<ваш_email@example.com>`
*   **LinkedIn:** `<ссылка_на_ваш_linkedin>`
