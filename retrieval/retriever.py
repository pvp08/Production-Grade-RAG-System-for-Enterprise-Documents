from langchain.vectorstores import FAISS
from ingestion.embeddings import get_embedding_model

def load_vectorstore(path="data/processed/faiss_index"):
    embeddings = get_embedding_model()
    return FAISS.load_local(path, embeddings)

def retrieve(query, k=5):
    vectorstore = load_vectorstore()
    results = vectorstore.similarity_search(query, k=k)
    return results

if __name__ == "__main__":
    docs = retrieve("What is machine learning?")
    for d in docs:
        print(d.page_content[:200])
