import mlflow
from retrieval.retriever import retrieve
from evaluation.data import EVAL_DATA

def recall_at_k(retrieved_docs, relevant_docs):
    retrieved_text = " ".join([d.page_content.lower() for d in retrieved_docs])
    for rel in relevant_docs:
        if rel.lower() in retrieved_text:
            return 1
    return 0

def evaluate(k=5):
    mlflow.set_experiment("rag-retrieval")
    recalls = []

    with mlflow.start_run():
        mlflow.log_param("k", k)

        for sample in EVAL_DATA:
            docs = retrieve(sample["question"], k=k)
            score = recall_at_k(docs, sample["relevant_docs"])
            recalls.append(score)

        avg_recall = sum(recalls) / len(recalls)
        mlflow.log_metric("recall_at_k", avg_recall)
        print(f"Recall@{k}: {avg_recall:.2f}")

if __name__ == "__main__":
    evaluate()
