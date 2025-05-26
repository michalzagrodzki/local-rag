from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
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
    """Embed & search the vector store for top-k chunks."""
    docs = db.similarity_search(question, k=k)
    # Combine chunks into a single context string
    context = "\n\n---\n\n".join(d.page_content for d in docs)
    return context, docs


def make_chain() -> LLMChain:
    """Construct an LLMChain with the retrieval prompt template."""
    template = (
        "Use ONLY the context below to answer the question.\n\n"
        "Context:\n{context}\n\n"
        "Question:\n{question}\n\n"
        "Answer:"
    )
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )
    llm = ChatOllama(
        model=OLLAMA_MODEL,
        ollama_api_url=OLLAMA_API_URL
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain


def answer_question(db: Chroma, chain: LLMChain, question: str, k: int = 3) -> tuple[str, list]:
    """Retrieve context, run the chain, and return the answer plus source docs."""
    context, docs = retrieve_context(db, question, k)
    response = chain.run({"context": context, "question": question})
    return response, docs
