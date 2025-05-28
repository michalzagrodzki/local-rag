from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from .config import (CHROMA_DB_DIR, 
                      EMBEDDING_MODEL, OLLAMA_MODEL, OLLAMA_API_URL, TOP_K, 
                      DEBUG, TEMPERATURE, TOP_P, LLM_TOP_K, MIN_P,
                      PRESENCE_PENALTY, MAX_OUTPUT_TOKENS,)


def load_index() -> Chroma:
    """Load the persisted ChromaDB index and embedding function."""
    embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = Chroma(
        persist_directory=str(CHROMA_DB_DIR),
        embedding_function=embedder
    )
    return db

def retrieve_context(db: Chroma, question: str, k: int = TOP_K) -> tuple[str, list]:
    docs = db.similarity_search(question, k=k)
    context = "\n\n---\n\n".join(d.page_content for d in docs)
    if DEBUG:
        print("\n[DEBUG] Retrieved top", k, "chunks:")
        for i, d in enumerate(docs, 1):
            print(f"--- chunk {i} ---\n{d.page_content[:300]}...\n")
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

# add sampling parameters
sample_kwargs = {
    "temperature":      TEMPERATURE,
    "top_p":            TOP_P,
    "top_k":            LLM_TOP_K,
    "min_p":            MIN_P,
    "presence_penalty": PRESENCE_PENALTY,
    "max_tokens":       MAX_OUTPUT_TOKENS,
}

llm = ChatOllama(
    model=OLLAMA_MODEL,
    ollama_api_url=OLLAMA_API_URL,
    **sample_kwargs
)

rag_chain = RunnableSequence(prompt | llm | StrOutputParser())

def answer_with_history(db, question, chat_history, k=TOP_K):
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

    prompt_inputs = {
      "chat_history": history_block,
      "context": context,
      "question": question
    }
    full_prompt = prompt.format(**prompt_inputs)
    if DEBUG:
        print("\n[DEBUG] Full prompt sent to LLM:\n", full_prompt, "\n")

    # 3. invoke the runnable
    answer = rag_chain.invoke(prompt_inputs)

    # 4. update history
    new_history = chat_history + [(question, answer)]
    return answer, new_history, docs