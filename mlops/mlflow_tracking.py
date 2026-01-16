import mlflow

def start_experiment(chunk_size, overlap, embedding_model):
    mlflow.set_experiment("rag-ingestion")
    with mlflow.start_run():
        mlflow.log_param("chunk_size", chunk_size)
        mlflow.log_param("chunk_overlap", overlap)
        mlflow.log_param("embedding_model", embedding_model)

