import threading
import itertools
import time
import sys
import re
from src.ingest import build_or_load_index
from src.qa import load_index, answer_with_history

def spinner(done_event):
    """Simple CLI spinner until done_event is set."""
    for c in itertools.cycle('|/-\\'):
        if done_event.is_set():
            break
        sys.stdout.write(f"\rThinking {c}")
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write("\r" + " " * 20 + "\r")

def main():
    db = build_or_load_index()

    print("\n🗨️  Retrieval-Augmented Generation ready. Ask anything!\n")
    chat_history = []
    while True:
        q = input("Your question (or ‘exit’): ").strip()
        if q.lower() in ("exit", "quit"):
            print("👋 Bye!")
            break
        
        done_event = threading.Event()
        spin_thread = threading.Thread(target=spinner, args=(done_event,))
        spin_thread.start()

        raw_answer, chat_history, sources = answer_with_history(db, q, chat_history, k=5)

        done_event.set()
        spin_thread.join()

        answer = re.sub(r'<think>.*?</think>', '', raw_answer, flags=re.DOTALL).strip()
        
        print("\n🤖 Answer:\n", answer, "\n")
        print("📑 Sources used:")
        for i, doc in enumerate(sources, 1):
            snippet = doc.page_content.replace("\n", " ")[:200]
            print(f"  {i}. (…){snippet}…")
        print("\n" + "-"*40 + "\n")

if __name__ == "__main__":
    main()