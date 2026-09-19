"""
Mini AI Knowledge Assistant (RAG)
----------------------------------
A Retrieval-Augmented Generation app that answers questions grounded in
user-uploaded documents (PDF, TXT, DOCX). Uses Groq for the LLM,
HuggingFace sentence-transformers for local embeddings, and FAISS as
the vector store.

Run with:
    streamlit run app.py
"""

import os
import tempfile

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(page_title="Mini AI Knowledge Assistant", page_icon="📚", layout="wide")
st.title("📚 Mini AI Knowledge Assistant (RAG)")
st.caption("Upload documents, ask questions, get answers grounded in your content — powered by Groq (Llama 3.3 70B).")

# --------------------------------------------------------------------------
# Sidebar: settings
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    groq_api_key = st.text_input(
        "Groq API Key",
        type="password",
        value=os.environ.get("GROQ_API_KEY", ""),
        help="Get a free key at https://console.groq.com/keys",
    )

    hf_api_token = st.text_input(
        "Hugging Face API Token",
        type="password",
        value=os.environ.get("HUGGINGFACEHUB_API_TOKEN", ""),
        help="Used for embeddings via the HF Inference API. Get a free token at https://huggingface.co/settings/tokens",
    )

    embedding_model = st.selectbox(
        "Embedding model (HF Inference API)",
        ["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-mpnet-base-v2", "BAAI/bge-base-en-v1.5"],
        index=0,
    )

    model_name = st.selectbox(
        "Groq model",
        ["openai/gpt-oss-120b", "qwen/qwen3.6-27b", "openai/gpt-oss-20b"],
        index=0,
    )

    chunk_size = st.slider("Chunk size", 200, 2000, 1000, step=100)
    chunk_overlap = st.slider("Chunk overlap", 0, 500, 150, step=50)
    top_k = st.slider("Chunks to retrieve (k)", 1, 10, 4)

    st.divider()
    st.markdown(
        "**Pipeline:** Load → Split → Embed (MiniLM, local) → "
        "Store (FAISS) → Retrieve → Generate (Groq LLM)"
    )

# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []


@st.cache_resource(show_spinner=False)
def get_embeddings(hf_api_token, embedding_model):
    return HuggingFaceEndpointEmbeddings(
        model=embedding_model,
        task="feature-extraction",
        huggingfacehub_api_token=hf_api_token,
    )


def load_document(uploaded_file):
    """Save the uploaded file to a temp path and load it with the right loader."""
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    if suffix == ".pdf":
        loader = PyPDFLoader(tmp_path)
    elif suffix == ".docx":
        loader = Docx2txtLoader(tmp_path)
    elif suffix == ".txt":
        loader = TextLoader(tmp_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    docs = loader.load()
    for d in docs:
        d.metadata["source"] = uploaded_file.name
    os.unlink(tmp_path)
    return docs


def build_vectorstore(uploaded_files, chunk_size, chunk_overlap, hf_api_token, embedding_model):
    all_docs = []
    for f in uploaded_files:
        all_docs.extend(load_document(f))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(all_docs)

    embeddings = get_embeddings(hf_api_token, embedding_model)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore, len(chunks)


SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the "
    "context below, which comes from the user's uploaded documents. "
    "If the answer isn't in the context, say you don't have enough "
    "information in the provided documents — do not make things up.\n\n"
    "Context:\n{context}"
)

QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
    ]
)


def build_qa_chain(vectorstore, groq_api_key, model_name, top_k):
    llm = ChatGroq(groq_api_key=groq_api_key, model_name=model_name, temperature=0)
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    question_answer_chain = create_stuff_documents_chain(llm, QA_PROMPT)
    chain = create_retrieval_chain(retriever, question_answer_chain)
    return chain


# --------------------------------------------------------------------------
# Document upload + processing
# --------------------------------------------------------------------------
st.subheader("1. Upload your documents")
uploaded_files = st.file_uploader(
    "PDF, DOCX, or TXT — you can upload more than one",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
)

col1, col2 = st.columns([1, 3])
with col1:
    process_clicked = st.button(
        "🔄 Process documents",
        type="primary",
        disabled=not uploaded_files or not hf_api_token,
    )

if not hf_api_token:
    st.caption("⚠️ Enter a Hugging Face API token in the sidebar to enable embedding (needed before processing).")

if process_clicked and uploaded_files:
    with st.spinner("Extracting text, chunking, and generating embeddings via the HF Inference API..."):
        try:
            vectorstore, n_chunks = build_vectorstore(
                uploaded_files, chunk_size, chunk_overlap, hf_api_token, embedding_model
            )
            st.session_state.vectorstore = vectorstore
            st.session_state.processed_files = [f.name for f in uploaded_files]
            st.success(f"Indexed {len(uploaded_files)} document(s) into {n_chunks} chunks.")
        except Exception as e:
            st.error(f"Failed to process documents: {e}")

if st.session_state.processed_files:
    st.info("Currently indexed: " + ", ".join(st.session_state.processed_files))

st.divider()

# --------------------------------------------------------------------------
# Q&A
# --------------------------------------------------------------------------
st.subheader("2. Ask a question")

if not groq_api_key:
    st.warning("Enter your Groq API key in the sidebar to ask questions.")
elif st.session_state.vectorstore is None:
    st.warning("Upload and process at least one document first.")
else:
    question = st.text_input("Your question", placeholder="e.g. What is the refund policy described in the document?")
    ask_clicked = st.button("Ask")

    if ask_clicked and question.strip():
        with st.spinner("Retrieving relevant chunks and generating an answer..."):
            try:
                chain = build_qa_chain(st.session_state.vectorstore, groq_api_key, model_name, top_k)
                result = chain.invoke({"input": question})
                answer = result["answer"]
                sources = result.get("context", [])

                st.session_state.chat_history.append((question, answer, sources))
            except Exception as e:
                st.error(f"Error generating answer: {e}")

    for q, a, sources in reversed(st.session_state.chat_history):
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            st.write(a)
            if sources:
                with st.expander(f"📄 Sources ({len(sources)} chunks)"):
                    for i, doc in enumerate(sources, 1):
                        src = doc.metadata.get("source", "unknown")
                        page = doc.metadata.get("page", None)
                        label = f"**{i}. {src}**" + (f" (page {page + 1})" if page is not None else "")
                        st.markdown(label)
                        st.caption(doc.page_content[:400] + ("..." if len(doc.page_content) > 400 else ""))
