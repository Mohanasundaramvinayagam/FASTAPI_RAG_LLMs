# ⚡ Low-Latency RAG & Cached LLM Inference Engine

A production-oriented Retrieval-Augmented Generation (RAG) API built with **FastAPI**, **ChromaDB**, **Hugging Face (`all-MiniLM-L6-v2`)**, **Groq LLM Inference**, **MD5 $O(1)$ Hash Caching**, and **SQLite / Prometheus Observability**.

## 🚀 Key Features & Architecture
1. **Automated Vector ETL Pipeline (`POST /ingest`):**
   - Extracts text page-by-page from uploaded PDFs using `pypdf`.
   - Encodes text into 384-dimensional dense vector embeddings using Hugging Face's `all-MiniLM-L6-v2`.
   - Upserts embeddings into **ChromaDB** for semantic similarity search.
2. **$O(1)$ Hash-Based Caching & RAG (`POST /query`):**
   - Normalizes and hashes incoming prompts using `MD5` for $O(1)$ in-memory cache lookups.
   - **Cache Hit Path (< 1 ms):** Returns cached responses in `~0.02 ms`, bypassing vector search and LLM token costs.
   - **Cache Miss Path (~400–700 ms):** Retrieves top relevant chunks from ChromaDB, injects context into the system prompt, and runs high-speed inference via the **Groq API**.
3. **SQL Telemetry & Prometheus Metrics (`GET /analytics` & `GET /metrics`):**
   - Logs every request (`query_text`, `cache_hit`, `latency_ms`, `created_at`) to a local **SQLite** database (`metrics.db`).
   - Computes real-time latency comparisons (`AVG`, `MIN`, `MAX`, `COUNT`) grouped by Cache Hit vs. LLM Inference.
   - Exposes `/metrics` via `prometheus-fastapi-instrumentator` for HTTP monitoring.

---

## 📊 Performance Benchmark

| Query Path | Source | Typical Latency | Speedup |
| :--- | :--- | :--- | :--- |
| **First Request (Cache Miss)** | ChromaDB + Groq LLM | `~450.00 ms` | Baseline (`1x`) |
| **Repeated Request (Cache Hit)** | In-Memory MD5 Hash Lookup | **`~0.02 ms`** | **`>99.9%` faster** |

---

## 🛠️ Local Setup & Installation

### 1. Clone the Repository & Create Virtual Environment
```bash
git clone [https://github.com/YOUR_USERNAME/low-latency-rag-engine.git](https://github.com/YOUR_USERNAME/low-latency-rag-engine.git)
cd low-latency-rag-engine
python -m venv venv
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Mac/Linux:
source venv/bin/activate