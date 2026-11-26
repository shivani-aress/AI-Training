"""Extract readable page text from PDFs and split it into text chunks."""

from pathlib import Path

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# Find documents next to this script, so it works from any current directory.
DOCUMENTS_FOLDER = Path(__file__).parent / "documents"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def extract_pdf_pages(pdf_path):
    """Return one record per page, with its text, filename, and page number."""
    extracted_pages = []

    try:
        reader = PdfReader(pdf_path)
    except Exception as error:
        # A damaged or unreadable PDF should not stop the other files from loading.
        print(f"Could not open {pdf_path.name}: {error}")
        return extracted_pages

    for page_number, page in enumerate(reader.pages, start=1):
        try:
            # Some pages contain no extractable text; store those as empty strings.
            page_text = page.extract_text() or ""
        except Exception as error:
            print(f"Could not extract page {page_number} of {pdf_path.name}: {error}")
            page_text = ""

        extracted_pages.append(
            {
                "text": page_text,
                "source": pdf_path.name,
                "page": page_number,
            }
        )

    return extracted_pages


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into chunks, keeping some characters between adjacent chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be zero or more and smaller than chunk_size")

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])

        # Stop after the last chunk instead of creating an extra overlap-only chunk.
        if end >= len(text):
            break

        # Move forward by the chunk length minus the overlap so text is repeated.
        start += chunk_size - overlap

    return chunks


def main():
    """Extract PDF pages, create chunks, and generate their embeddings."""
    pdf_files = sorted(DOCUMENTS_FOLDER.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in: {DOCUMENTS_FOLDER}")

    all_pages = []
    for pdf_path in pdf_files:
        all_pages.extend(extract_pdf_pages(pdf_path))

    # Create a chunk for each portion of each page and keep its page metadata.
    all_chunks = []
    for page in all_pages:
        for text_chunk in chunk_text(page["text"]):
            all_chunks.append(
                {
                    "text": text_chunk,
                    "source": page["source"],
                    "page": page["page"],
                }
            )

    print(f"Total chunks: {len(all_chunks)}")

    if all_chunks:
        # Load the pretrained model once, then encode every chunk's text.
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        chunk_texts = [chunk["text"] for chunk in all_chunks]
        embeddings = model.encode(chunk_texts)

        # Attach each generated vector to the chunk it belongs to.
        for chunk, embedding in zip(all_chunks, embeddings):
            chunk["embedding"] = embedding.tolist()

        print(f"Total embeddings generated: {len(embeddings)}")
        print(f"Embedding dimensions: {len(all_chunks[0]['embedding'])}")

        # Show one chunk and only the first ten values of its embedding.
        sample_chunk = all_chunks[0]
        print("\n--- Sample embedded chunk ---")
        print(f"Source: {sample_chunk['source']} | Page: {sample_chunk['page']}")
        print(f"Text: {sample_chunk['text']}")
        print(f"First 10 embedding values: {sample_chunk['embedding'][:10]}")
    else:
        print("Total embeddings generated: 0")
        print("Embedding dimensions: unavailable because there are no chunks")

    if not all_chunks:
        print("No text chunks were generated. The PDF pages may be empty or scanned images.")

    # Open the persistent database and get the collection, creating it if needed.
    # Run this script from the RAG_Project folder so ./chroma_db is the intended folder.
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="company_documents")

    print(f"Total chunks being stored: {len(all_chunks)}")

    if all_chunks:
        ids = []
        chunk_positions = {}

        # Build stable IDs from the PDF name, page, and chunk position.
        for chunk in all_chunks:
            page_key = (chunk["source"], chunk["page"])
            position = chunk_positions.get(page_key, 0) + 1
            chunk_positions[page_key] = position
            ids.append(f"{chunk['source']}_page_{chunk['page']}_chunk_{position}")

        # Upsert updates existing IDs on later runs instead of adding duplicates.
        collection.upsert(
            ids=ids,
            documents=[chunk["text"] for chunk in all_chunks],
            embeddings=[chunk["embedding"] for chunk in all_chunks],
            metadatas=[
                {"source": chunk["source"], "page": chunk["page"]}
                for chunk in all_chunks
            ],
        )
        print("Chunks and embeddings stored successfully.")
    else:
        print("No chunks to store.")

    print(f"Total records in collection: {collection.count()}")


if __name__ == "__main__":
    main()

