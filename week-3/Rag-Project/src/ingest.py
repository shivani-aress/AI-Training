"""Ingest policy PDFs into the persistent ChromaDB collection."""
import argparse
import sys
from .document_loader import DOCUMENTS_DIR, load_pdf_pages
from .embeddings import embed_texts, load_embedding_model
from .text_chunker import split_pages
from .vector_store import add_documents, get_collection

def ingest(rebuild: bool = False) -> tuple[int, int, int]:
    """Load, split, embed, store; return PDF, non-empty page, and chunk counts."""
    pages = load_pdf_pages()
    if not pages:
        print(f"No PDF pages with text found in {DOCUMENTS_DIR}. Add PDF files and try again.")
        return 0, 0, 0
    pdf_count = len({page["metadata"]["source"] for page in pages})
    chunks = split_pages(pages)
    model = load_embedding_model()
    embeddings = embed_texts(model, [chunk["text"] for chunk in chunks])
    collection = get_collection(rebuild=rebuild)
    add_documents(collection, chunks, embeddings)
    print(f"PDF files processed: {pdf_count}")
    print(f"Non-empty pages processed: {len(pages)}")
    print(f"Chunks processed: {len(chunks)}")
    print(f"Chunks in collection: {collection.count()}")
    return pdf_count, len(pages), len(chunks)

def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest policy PDFs into ChromaDB.")
    parser.add_argument("--rebuild", action="store_true", help="Recreate the collection before ingesting all PDFs.")
    args = parser.parse_args()
    try:
        ingest(rebuild=args.rebuild)
    except Exception as exc:
        print(f"Ingestion failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

if __name__ == "__main__":
    main()
