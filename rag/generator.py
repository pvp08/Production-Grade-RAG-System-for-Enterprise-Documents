from langchain_groq import ChatGroq
from rag.prompt import RAG_PROMPT

llm = ChatGroq(
    model="gemma-7b",
    temperature=0
)

def generate_answer(question, docs):
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = RAG_PROMPT.format(
        context=context,
        question=question
    )

    response = llm.invoke(prompt)
    return response.content
