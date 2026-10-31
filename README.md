# ⚡ Full-Stack RAG & Low-Latency LLM Inference Engine

A decoupled, containerized AI microservice application pairing an asynchronous **FastAPI** backend with an interactive **Streamlit** frontend. Built to solve high LLM latency and redundant token costs through **Vector RAG**, **$O(1)$ Hash-Based Prompt Caching**, and **Relational SQL Telemetry**.

---

## 🏗️ System Architecture & Core Workflow

1. **Interactive Client Layer (`Streamlit`):**
   - Provides a clean UI for PDF document uploads, natural language querying, and real-time visualization of cache vs. LLM latency metrics.
2. **Asynchronous API Gateway (`FastAPI`):**
   - Exposes RESTful endpoints (`/ingest`, `/query`, `/analytics`, `/metrics`) with Pydantic schema validation and auto-generated Swagger UI documentation.
3. **Automated Document ETL & Vector Search (`ChromaDB` + `Sentence-Transformers`):**
   - Extracts raw text from PDFs (`pypdf`), encodes chunks into 384-dimensional dense vector embeddings (`all-MiniLM-L6-v2`), and performs semantic similarity retrieval.
4. **$O(1)$ Prompt Caching Layer (`MD5 Hash Lookup`):**
   - Normalizes incoming queries and computes an MD5 digest prior to vector search.
   - **Cache Hit (<2ms):** Returns pre-computed responses instantly, cutting latency by **99%** and saving LLM API compute.
   - **Cache Miss (~400ms):** Retrieves top context chunks from ChromaDB and invokes **Llama-3.1-8B** via Groq.
5. **Production Observability (`SQLite` + `Prometheus`):**
   - Logs every request's execution time and cache status into a relational SQL database, executing `GROUP BY` aggregations (`AVG`, `MIN`, `MAX`, `COUNT`) to monitor SLAs alongside `/metrics` for Prometheus scraping.

---

## 🛠️ Tech Stack
- **Backend & API:** Python, FastAPI, Uvicorn, Pydantic
- **Frontend UI:** Streamlit, Requests
- **AI & Vector Pipeline:** Llama-3.1-8B (Groq), ChromaDB, Hugging Face `sentence-transformers`, `pypdf`
- **Database & Telemetry:** SQLite3 (Relational SQL Aggregations), Prometheus (`prometheus-fastapi-instrumentator`)
- **DevOps & Packaging:** Docker

---

## 🚀 Quick Start (Run Locally)

### 1. Clone the Repository & Install Dependencies
```bash
git clone [https://github.com/Mohanasundaramvinayagam/FASTAPI_RAG_LLMs.git](https://github.com/Mohanasundaramvinayagam/FASTAPI_RAG_LLMs.git)
cd FASTAPI_RAG_LLMs
pip install -r requirements.txt