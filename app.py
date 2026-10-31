import os
import time
import sqlite3
import hashlib
import traceback
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from groq import Groq
from prometheus_fastapi_instrumentator import Instrumentator

# Initialize FastAPI App
app = FastAPI(
    title="Low-Latency RAG & Cached LLM Inference Engine",
    description="Production MLOps pipeline with Vector ETL, Hash Caching, and SQL Telemetry"
)

# 1. Expose Prometheus Metrics at /metrics
Instrumentator().instrument(app).expose(app)

# 2. Initialize Vector Database (ChromaDB) & Embedding Model
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="knowledge_base")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# 3. Initialize In-Memory Cache & SQLite Telemetry Database
prompt_cache = {}
conn = sqlite3.connect("metrics.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS query_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_text TEXT,
    cache_hit INTEGER,
    latency_ms REAL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")
conn.commit()
load_dotenv()

# 4. Initialize Groq Client (Loads safely from .env)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
groq_client = Groq(api_key=GROQ_API_KEY)

class QueryRequest(BaseModel):
    prompt: str

@app.post("/ingest", summary="1. Upload PDF to Vector Database")
async def ingest_pdf(file: UploadFile = File(...)):
    try:
        reader = PdfReader(file.file)
        chunks, ids = [], []
        
        for idx, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                chunks.append(text.strip())
                ids.append(f"{file.filename}_p{idx}")
                
        if not chunks:
            return {"error": "No readable text found in PDF."}

        embeddings = embedder.encode(chunks).tolist()
        collection.upsert(documents=chunks, embeddings=embeddings, ids=ids)
        return {
            "status": "Successfully Ingested",
            "filename": file.filename,
            "pages_indexed": len(chunks)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingest Error: {str(e)}")

@app.post("/query", summary="2. Ask Questions with O(1) Cache & RAG")
async def query_llm(request: QueryRequest):
    try:
        if not GROQ_API_KEY.startswith("gsk_"):
            raise ValueError("Invalid Groq API Key! Make sure you pasted your 'gsk_...' key on Line 41 of app.py and saved the file.")

        start_time = time.perf_counter()
        
        # Normalize and hash the prompt for O(1) cache lookup
        clean_prompt = request.prompt.strip().lower()
        prompt_hash = hashlib.md5(clean_prompt.encode()).hexdigest()

        # PATH A: CACHE HIT (Returns in < 2ms)
        if prompt_hash in prompt_cache:
            latency = round((time.perf_counter() - start_time) * 1000, 2)
            cursor.execute(
                "INSERT INTO query_logs (query_text, cache_hit, latency_ms) VALUES (?, ?, ?)",
                (request.prompt, 1, latency)
            )
            conn.commit()
            return {
                "answer": prompt_cache[prompt_hash],
                "source": "CACHE_HIT (O(1) Lookup)",
                "latency_ms": latency
            }

        # PATH B: CACHE MISS -> Run Vector Search + Llama-3 Inference
        doc_count = collection.count()
        if doc_count > 0:
            query_vec = embedder.encode([request.prompt]).tolist()
            results = collection.query(query_embeddings=query_vec, n_results=min(2, doc_count))
            context = "\n---\n".join(results["documents"][0]) if results["documents"] and results["documents"][0] else "No context uploaded yet."
        else:
            context = "No PDF uploaded yet. Answer based on general knowledge."

        completion = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": f"You are a helpful AI assistant. Answer the user's question strictly using the provided context:\n{context}"},
                {"role": "user", "content": request.prompt}
            ],
            temperature=0.2
        )
        answer = completion.choices[0].message.content

        # Store in cache and log to SQL
        prompt_cache[prompt_hash] = answer
        latency = round((time.perf_counter() - start_time) * 1000, 2)
        cursor.execute(
            "INSERT INTO query_logs (query_text, cache_hit, latency_ms) VALUES (?, ?, ?)",
            (request.prompt, 0, latency)
        )
        conn.commit()

        return {
            "answer": answer,
            "source": "LLM_RAG_INFERENCE (Llama-3.1-8B)",
            "latency_ms": latency
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query Error: {str(e)} | Trace: {traceback.format_exc().splitlines()[-1]}")

@app.get("/analytics", summary="3. View SQL Latency & Cache Telemetry")
async def get_sql_metrics():
    cursor.execute("""
        SELECT 
            CASE WHEN cache_hit = 1 THEN 'Cache Hit' ELSE 'LLM Inference (Miss)' END as request_type,
            COUNT(*) as total_requests,
            ROUND(AVG(latency_ms), 2) as avg_latency_ms,
            ROUND(MIN(latency_ms), 2) as min_latency_ms,
            ROUND(MAX(latency_ms), 2) as max_latency_ms
        FROM query_logs
        GROUP BY cache_hit
    """)
    rows = cursor.fetchall()
    return {
        "telemetry_summary": [
            {
                "type": r[0],
                "count": r[1],
                "avg_latency_ms": r[2],
                "min_latency_ms": r[3],
                "max_latency_ms": r[4]
            } for r in rows
        ]
    }