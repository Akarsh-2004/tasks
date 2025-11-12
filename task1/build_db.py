from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings  # ✅ updated package

DB_DIR = "db"

def build_db():
    pdf_paths = ["2506.02153v2.pdf", "reasoning_models_paper.pdf"]
    documents = []
    for path in pdf_paths:
        print(f"Loading: {path}")
        loader = PyPDFLoader(path)
        documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR,
        collection_name="pdf_chunks"
    )

    db.persist()
    print("✅ Vector DB created successfully.")

if __name__ == "__main__":
    build_db()
