import streamlit as st
import requests
import pandas as pd

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Low-Latency RAG Engine",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Low-Latency RAG & Cached LLM Inference Engine")
st.caption("FastAPI Backend • ChromaDB Vector Search • O(1) MD5 Hash Cache • SQLite Telemetry")

# --- SIDEBAR: 1. PDF Ingestion & 3. SQL Telemetry ---
with st.sidebar:
    st.header("📄 1. Upload Knowledge Base")
    uploaded_file = st.file_uploader("Upload a PDF Document", type=["pdf"])
    
    if st.button("Ingest PDF to Vector DB", use_container_width=True):
        if uploaded_file is not None:
            with st.spinner("Extracting text & indexing vectors..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    res = requests.post(f"{API_URL}/ingest", files=files)
                    if res.status_code == 200:
                        data = res.json()
                        if "error" in data:
                            st.error(data["error"])
                        else:
                            st.success(f"Indexed {data['pages_indexed']} page(s) from `{data['filename']}`!")
                    else:
                        st.error(f"Error: {res.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to FastAPI! Make sure `uvicorn app:app --reload` is running.")
        else:
            st.warning("Please select a PDF file first.")

    st.divider()

    st.header("📊 3. Live SQL Telemetry")
    if st.button("Refresh SQL Metrics", use_container_width=True):
        try:
            res = requests.get(f"{API_URL}/analytics")
            if res.status_code == 200:
                summary = res.json().get("telemetry_summary", [])
                if summary:
                    df = pd.DataFrame(summary)
                    df.columns = ["Type", "Requests", "Avg (ms)", "Min (ms)", "Max (ms)"]
                    st.dataframe(df, use_container_width=True, hide_index=True)
                else:
                    st.info("No queries logged yet.")
        except requests.exceptions.ConnectionError:
            st.error("FastAPI backend is offline.")

# --- MAIN AREA: 2. Chat Interface with Latency Badges ---
st.subheader("💬 2. Ask Your Document (O(1) Cache vs. RAG Inference)")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "meta" in msg:
            st.caption(msg["meta"])

# Chat Input
if prompt := st.chat_input("Ask a question about your uploaded PDF..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call FastAPI /query endpoint
    with st.chat_message("assistant"):
        with st.spinner("Querying engine..."):
            try:
                res = requests.post(f"{API_URL}/query", json={"prompt": prompt})
                if res.status_code == 200:
                    data = res.json()
                    answer = data["answer"]
                    source = data["source"]
                    latency = data["latency_ms"]

                    # Visual badge for Cache Hit vs LLM Inference
                    if "CACHE_HIT" in source:
                        badge = f"🟢 **{source}** | ⚡ Latency: **`{latency} ms`**"
                    else:
                        badge = f"🟠 **{source}** | ⏱️ Latency: **`{latency} ms`**"

                    st.markdown(answer)
                    st.caption(badge)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "meta": badge
                    })
                else:
                    st.error(f"Backend Error: {res.text}")
            except requests.exceptions.ConnectionError:
                st.error("⚠️ FastAPI server is not running! Start `uvicorn app:app --reload` in Terminal 1.")