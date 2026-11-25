"""Create and manage the persistent ChromaDB collection."""
import hashlib
from pathlib import Path
import chromadb

PROJECT_DIR = Path(__file__).resolve().parents[1]
CHROMA_DIR = PROJECT_DIR / "chroma_db"
COLLECTION_NAME = "internal_company_policies"

def get_collection(rebuild: bool = False, persist_dir: Path = CHROMA_DIR):
    """Open the persistent cosine-distance collection, optionally recreating it."""
    client = chromadb.PersistentClient(path=str(persist_dir))
    if rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception as exc:
            if "does not exist" not in str(exc).lower():
                raise
    return client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

def make_chunk_id(chunk: dict) -> str:
    """Build a stable ID to avoid duplicate chunks when ingestion repeats."""
    value = f"{chunk['metadata']['source']}|{chunk['metadata']['page']}|{chunk['text']}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()

def add_documents(collection, chunks: list[dict], embeddings: list[list[float]]) -> int:
    """Upsert chunk text, embeddings, and metadata; return number of chunks."""
    if len(chunks) != len(embeddings):
        raise ValueError("Each chunk must have exactly one embedding.")
    if not chunks:
        return 0
    collection.upsert(
        ids=[make_chunk_id(chunk) for chunk in chunks],
        documents=[chunk["text"] for chunk in chunks],
        embeddings=embeddings,
        metadatas=[chunk["metadata"] for chunk in chunks],
    )
    return len(chunks)

def get_document_count(collection) -> int:
    """Return the number of stored chunks."""
    return collection.count()
