from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader

def load_documents(data_path="data/raw"):
    documents = []
    for file in Path(data_path).glob("*"):
        if file.suffix==".pdf":
            loader=PyPDFLoader(str(file))
        elif file.suffix==".txt":
            loader = TextLoader(str(file))
        else:
            continue
        documents.extend(loader.load())
    return documents

if __name__ == "__main__":
    docs=load_documents
    print(f"Loaded{len(docs)} documents")
