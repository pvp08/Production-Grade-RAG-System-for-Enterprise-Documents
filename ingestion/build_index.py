
import faiss
from langchain_community.vectorstores import FAISS
from load_docs import load_documents
from chunking import chunk_documents
from embeddings import get_embedding_model

def build_faiss_index():
    docs = load_documents()
    chunks = chunk_documents(docs)
    embeddings = get_embedding_model()
    vectorstore = FAISS.from_documents(chunks,embeddings)
    vectorstore.save_local("data/processed/faiss_index")
    print("FAISS index saved successfully")

    if __name__ == "__main__":
        build_faiss_index()