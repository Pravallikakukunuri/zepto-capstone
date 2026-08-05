"""
ingest.py
Loads the 8 Zepto policy documents from docs/, embeds each one (one chunk
per document, since each document is short) using the open-source
all-MiniLM-L6-v2 model via sentence-transformers, and stores the embeddings
in a persistent ChromaDB collection on disk.

Run this ONCE before starting the FastAPI app:
    python ingest.py

Re-running it is safe -- it clears and rebuilds the collection each time.
"""

import os
import glob
from sentence_transformers import SentenceTransformer
import chromadb

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "docs")
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "zepto_policies"


def load_documents():
    paths = sorted(glob.glob(os.path.join(DOCS_DIR, "doc_*.txt")))
    docs = []
    for path in paths:
        doc_id = os.path.splitext(os.path.basename(path))[0]  # e.g. "doc_01"
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        docs.append({"id": doc_id, "text": text})
    return docs


def main():
    docs = load_documents()
    print(f"Loaded {len(docs)} documents from {DOCS_DIR}")
    for d in docs:
        print(f"  {d['id']}: {d['text'][:60]}...")

    print("\nLoading embedding model (all-MiniLM-L6-v2) -- this downloads "
          "the model the first time, then it is cached locally...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    texts = [d["text"] for d in docs]
    ids = [d["id"] for d in docs]
    print("Encoding documents...")
    embeddings = embedder.encode(texts).tolist()

    print(f"\nWriting to ChromaDB at {CHROMA_PATH} ...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    collection.add(ids=ids, documents=texts, embeddings=embeddings)
    print(f"Stored {len(ids)} chunks in collection '{COLLECTION_NAME}'.")
    print("\nDone. You can now run: python main.py  (or) uvicorn main:app --reload")


if __name__ == "__main__":
    main()