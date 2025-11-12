import os
import time
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM  # updated package for LangChain 1.0+

# Load environment variables (if any)
load_dotenv()

DB_DIR = "db"

st.set_page_config(page_title="🔍 Local PDF Q&A", layout="wide")
st.title("🔍 Local PDF Q&A — MiniLM-L12 + Gemma2:2B")

# -----------------------------
# Load or create vector database
# -----------------------------
@st.cache_resource
def load_db():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L12-v2")  # 384D, better than L6

    return Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings,
        collection_name="pdf_chunks"
    )

db = load_db()

# -----------------------------
# LLM: Gemma2 2B (lightweight)
# -----------------------------
llm = OllamaLLM(model="gemma2:2b")

# -----------------------------
# User input
# -----------------------------
question = st.text_input("Ask your question:")
run = st.button("Search")

if run and question:
    start = time.time()

    # Search top 5 relevant chunks
    docs = db.similarity_search(question, k=5)
    context = "\n\n".join([d.page_content for d in docs])

    # Prompt — explicit about synthesizing multiple chunks
    prompt = f"""
You are a helpful AI assistant.
Use ONLY the context below to answer the question.
You can summarize, synthesize, or combine info from multiple chunks.
If the answer is not in the context, say: "Not found in the document."

Context:
{context}

Question: {question}

Answer:
"""

    # Generate answer
    answer = llm.invoke(prompt)
    end = time.time()

    # -----------------------------
    # Display results
    # -----------------------------
    st.subheader("Answer")
    st.write(answer.content if hasattr(answer, "content") else answer)

    st.subheader("Context Used")
    for i, d in enumerate(docs):
        with st.expander(f"Chunk {i+1}"):
            st.write(d.page_content)

    st.info(f"⏱ Time: {round(end-start, 3)} sec")
