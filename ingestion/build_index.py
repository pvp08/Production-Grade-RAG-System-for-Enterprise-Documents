from ingestion.load_docs import load_documents
from ingestion.chunking import chunk_documents
from ingestion.embeddings import get_embedding_model
from langchain_community.vectorstores import FAISS
from pathlib import Path

def build_faiss_index():
    docs = load_documents()
    chunks = chunk_documents(docs)

    embeddings = get_embedding_model()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    output_path = Path("data/processed/faiss_index")
    output_path.mkdir(parents=True, exist_ok=True)

    vectorstore.save_local(output_path)
    print("FAISS index saved at:", output_path)

if __name__ == "__main__":
    build_faiss_index()
