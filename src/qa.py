from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from .config import (CHROMA_DB_DIR, EMBEDDING_MODEL, OLLAMA_MODEL, OLLAMA_API_URL)


def load_index() -> Chroma:
    """Load the persisted ChromaDB index and embedding function."""
    embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = Chroma(
        persist_directory=str(CHROMA_DB_DIR),
        embedding_function=embedder
    )
    return db

def retrieve_context(db: Chroma, question: str, k: int = 3) -> tuple[str, list]:
    docs = db.similarity_search(question, k=k)
    context = "\n\n---\n\n".join(d.page_content for d in docs)
    return context, docs

template = (
    "You are a helpful assistant. Keep in mind the conversation so far:\n"
    "{chat_history}\n\n"
    "Now use ONLY the context below to answer the new question.\n\n"
    "Context:\n{context}\n\n"
    "Question:\n{question}\n\n"
    "Answer:"
)

prompt = PromptTemplate(
    input_variables=["chat_history", "context", "question"],
    template=template
)

llm = ChatOllama(
    model=OLLAMA_MODEL,
    ollama_api_url=OLLAMA_API_URL
)

rag_chain = RunnableSequence(prompt | llm | StrOutputParser())

def answer_with_history(db, question, chat_history, k=3):
    """
    Returns: answer_text (str), updated_history (list), source_docs (list)
    """
    # 1. retrieve fresh context
    context, docs = retrieve_context(db, question, k=k)

    # 2. turn chat_history into a single block
    if chat_history:
        history_block = "\n".join(
            f"User: {u}\nAssistant: {a}" for u, a in chat_history
        )
    else:
        history_block = "(none)\n"

    # 3. invoke the runnable
    answer = rag_chain.invoke({
        "chat_history": history_block,
        "context": context,
        "question": question
    })

    # 4. update history
    new_history = chat_history + [(question, answer)]
    return answer, new_history, docs