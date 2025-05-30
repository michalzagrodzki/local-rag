from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from .config import PDF_DIR, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL, CHROMA_DB_DIR, DEBUG

HNSW_CONFIG = {
    "hnsw:space": "cosine",         # use cosine distance
    "hnsw:construction_ef": 100,    # neighbors to explore during construction
    "hnsw:M": 16,                   # maximum neighbor connections
    "hnsw:search_ef": 10,           # neighbors to explore during search
    "hnsw:num_threads": 4,          # threads for HNSW operations
    "hnsw:resize_factor": 1.2,      # growth factor when resizing
    "hnsw:batch_size": 100,         # brute-force batch size fallback threshold
    "hnsw:sync_threshold": 1000,    # threshold to write HNSW index to disk
}

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
        chunks = splitter.split_documents(pages)
        if DEBUG:
            print(f"[DEBUG] {pdf.name}: split into {len(chunks)} chunks.")
        docs.extend(chunks)
    return docs

def build_index(docs):
    embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    index = Chroma.from_documents(
        docs,
        embedder,
        persist_directory=str(CHROMA_DB_DIR),
        collection_metadata=HNSW_CONFIG,
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
            embedding_function=embedder,
            collection_metadata=HNSW_CONFIG,
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