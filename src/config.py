from pathlib import Path

# Document folder
PDF_DIR           = Path(__file__).parent.parent / "data"

# Embedding model (HuggingFace or Ollama)
EMBEDDING_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"

# Vector store options
CHROMA_DB_DIR     = Path(__file__).parent.parent / "chroma_db"

# How many chunks to feed into the prompt by default
TOP_K             = 5

# Toggle detailed logging of context + prompt
DEBUG             = True

# Chunking parameters
CHUNK_SIZE       = 1000    # characters per chunk
CHUNK_OVERLAP    = 200     # characters overlap

OLLAMA_MODEL      = "qwen3:30b"               # or your chosen model
OLLAMA_API_URL    = "http://localhost:11434"  # default Ollama HTTP endpoint

# LLM parameters
TEMPERATURE       = 0.7
TOP_P             = 0.8
LLM_TOP_K         = 20
MIN_P             = 0.0

# Presence penalty (0–2)
PRESENCE_PENALTY  = 0.0

# Max output length (in tokens)
MAX_OUTPUT_TOKENS = 32768