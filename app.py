from rag.rag_pipeline import rag_answer

if __name__ == "__main__":
    while True:
        question = input("\nAsk a question (or type 'exit'): ")
        if question.lower() == "exit":
            break

        answer, docs = rag_answer(question)
        print("\nAnswer:\n", answer)

        print("\nSources:")
        for i, doc in enumerate(docs):
            print(f"{i+1}. {doc.metadata.get('source')}")
