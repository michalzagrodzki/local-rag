from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from .config import PDF_DIR, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL, VECTOR_STORE, CHROMA_DB_DIR

def load_and_split() -> list:
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    docs = []
    for pdf in PDF_DIR.glob("*.pdf"):
        loader = PyPDFLoader(str(pdf))
        pages = loader.load_and_split()
        docs.extend(splitter.split_documents(pages))
    return docs

def build_index(docs):
    embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    index = Chroma.from_documents(
        docs,
        embedder,
        persist_directory=str(CHROMA_DB_DIR),
    )
    index.persist()
    return index

def build_or_load_index():
    """
    If the ChromaDB folder exists and has data, load it.
    Otherwise, rebuild it from PDFs.
    """
    try:
        # Try to load an existing index
        embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        index = Chroma(
            persist_directory=str(CHROMA_DB_DIR),
            embedding_function=embedder
        )
        # Quick test to see if it has any vectors
        if not index._collection.count():
            raise ValueError("Empty index, rebuilding.")
        print("✅ Loaded existing ChromaDB index.")
    except Exception:
        # On failure, rebuild from scratch
        print("🔄 No valid index found; rebuilding from PDFs...")
        docs = load_and_split()
        index = build_index(docs)
    return index