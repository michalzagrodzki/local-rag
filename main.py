import time
import gradio as gr
import re

from src.ingest import build_or_load_index
from src.qa import answer_with_history
db = build_or_load_index()

def chat_stream(user_message, chat_history):
    """
    Gradio will call this generator to update the chat.
    We:
      a) run RAG+LLM once to get the full answer,
      b) strip out any <think>...</think> sections,
      c) then stream it back word by word.
    """
    # ensure there's a list
    messages = chat_history or []

    messages = messages + [{"role": "user", "content": user_message}]

    # a) get the raw answer (with full pipeline + history)
    tuple_history = [(m["content"], "") for m in messages if m["role"]=="user"]  # keep only user turns
    raw_answer, new_history, sources = answer_with_history(db, user_message, tuple_history, k=5)

    # b) remove any <think>…</think> chains of thought
    clean_answer = re.sub(
        r'<think>.*?</think>',
        '',
        raw_answer,
        flags=re.DOTALL
    ).strip()

    # c) stream word-by-word
    words = clean_answer.split()
    partial = ""
    for w in words:
        partial += w + " "
        yield messages + [{"role": "assistant", "content": partial.strip()}]
        time.sleep(0.05)

    # finally, show sources in console or log (optional)
    print("Sources for this turn:")
    for doc in sources:
        print(" -", doc.metadata.get("source", "unknown"))

def main():
    with gr.Blocks() as demo:
        gr.Markdown("## 📚 PDF Chatbot (Local RAG)\nStreaming + follow-ups supported.")
        # Use messages format to avoid deprecation warning
        chatbot = gr.Chatbot(type="messages", elem_id="chatbot")
        user_input = gr.Textbox(placeholder="Type your question…", show_label=False)

        user_input.submit(
            fn=chat_stream,
            inputs=[user_input, chatbot],
            outputs=chatbot
        )

    demo.launch()

if __name__ == "__main__":
    main()