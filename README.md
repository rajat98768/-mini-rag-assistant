
# Mini AI Knowledge Assistant (RAG)

A simple Retrieval-Augmented Generation app: upload documents, ask questions,
get answers grounded in that content. Built with Streamlit, LangChain, FAISS,
local HuggingFace embeddings, and **Groq** as the LLM backend.

## Pipeline

1. **Load** — PDF / DOCX / TXT files via LangChain document loaders
2. **Split** — `RecursiveCharacterTextSplitter` (configurable chunk size/overlap)
3. **Embed** — Hugging Face Inference API (`sentence-transformers/all-MiniLM-L6-v2` by default, swappable in the sidebar) — no local model download, runs via API call
4. **Store** — FAISS in-memory vector store
5. **Retrieve** — top-k similarity search per question
6. **Generate** — Groq-hosted `openai/gpt-oss-120b` answers using only the retrieved context (model picker in the sidebar also offers `qwen/qwen3.6-27b` and `openai/gpt-oss-20b`)

## Setup

```bash
pip install -r requirements.txt
```

You need two free API keys:

- **Groq** (for the LLM): https://console.groq.com/keys
- **Hugging Face** (for embeddings via the Inference API): https://huggingface.co/settings/tokens

Paste both into the sidebar at runtime, or set them as environment variables
before launching:

```bash
export GROQ_API_KEY="your-groq-key-here"                # macOS/Linux
export HUGGINGFACEHUB_API_TOKEN="your-hf-token-here"     # macOS/Linux
set GROQ_API_KEY=your-groq-key-here                      # Windows (cmd)
set HUGGINGFACEHUB_API_TOKEN=your-hf-token-here          # Windows (cmd)
```

## Run

```bash
streamlit run app.py
```

Then in the browser tab that opens:

1. Upload one or more PDF / DOCX / TXT files in **"1. Upload your documents"**
2. Click **"Process documents"** — this chunks the text and calls the HF Inference API to embed each chunk into FAISS
3. Type a question in **"2. Ask a question"** and click **Ask**
4. Expand **"Sources"** under any answer to see which document chunks it was grounded in

## Notes

- Answers are constrained by a prompt instructing the model to rely only on
  retrieved context and to say so when the documents don't contain the answer
  (reduces hallucination, satisfies requirement #8).
- Both embeddings and generation now go through hosted APIs — no local model
  weights are downloaded, so first-run and low-RAM environments are faster.
- The free HF Inference API can cold-start (a few seconds' delay) or
  occasionally rate-limit on the first call to a model; a retry after a few
  seconds usually works.
- To swap in a persistent vector store (e.g. for large document sets across
  sessions), replace `FAISS.from_documents(...)` with a Chroma
  `persist_directory`, or a Pinecone index — the rest of the pipeline is unchanged.
- Adjust chunk size/overlap and `k` (chunks retrieved) in the sidebar to trade
  off answer precision vs. context breadth.

## Tech stack

| Stage        | Tool                                                |
|--------------|------------------------------------------------------|
| UI           | Streamlit                                             |
| Orchestration| LangChain                                             |
| Embeddings   | Hugging Face Inference API (`all-MiniLM-L6-v2`, swappable) |
| Vector store | FAISS                                                 |
| LLM          | Groq (`openai/gpt-oss-120b`)                          |
<img width="3191" height="1835" alt="Screenshot 2026-09-19 205305" src="https://github.com/user-attachments/assets/7dbb48a7-b865-4b5f-89d4-f0bae8e48bf1" />
<img width="3186" height="1821" alt="Screenshot 2026-09-19 205316" src="https://github.com/user-attachments/assets/b868b264-fa39-4665-9299-9cba0e15f859" />

