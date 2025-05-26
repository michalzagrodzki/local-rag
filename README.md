# RAG-PDF-Chat
Tools:
- **LangChain** for orchestration
- **PyPDF** to load PDFs
- **ChromaDb** for vector indexing
- **Sentence-Transformers** for embeddings
- **Ollama** models (DeepSeek-R1, Qwen)
- **Gradio** UI

## Setup
1. `python3 -m venv .venv && source .venv/bin/activate`  
2. `pip install -r requirements.txt`  
3. `ollama pull deepseek-r1 qwen`  
4. `python main.py` 