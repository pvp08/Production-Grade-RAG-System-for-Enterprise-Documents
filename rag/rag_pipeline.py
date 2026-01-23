from retrieval.retriever import retrieve
from rag.generator import generate_answer

def rag_answer(question, k=5):
    docs = retrieve(question, k=k)
    answer = generate_answer(question, docs)
    return answer, docs
