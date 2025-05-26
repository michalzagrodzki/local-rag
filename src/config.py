from pathlib import Path

# Document folder
PDF_DIR = Path(__file__).parent.parent / "data"

# Embedding model (HuggingFace or Ollama)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# or if using Ollama:
# EMBEDDING_MODEL = OllamaEmbeddings(model="deepseek-r1")

# Vector store options
VECTOR_STORE = "chromadb"     # or "chromadb"
CHROMA_DB_DIR   = Path(__file__).parent.parent / "chroma_db"

# Chunking parameters
CHUNK_SIZE       = 1000    # characters per chunk
CHUNK_OVERLAP    = 200     # characters overlap

OLLAMA_MODEL      = "qwen3:30b"                 # or your chosen model
OLLAMA_API_URL    = "http://localhost:11434"  # default Ollama HTTP endpoint