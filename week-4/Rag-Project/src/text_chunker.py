"""Split each PDF page independently while carrying its source metadata."""
from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_pages(pages: list[dict], chunk_size: int = 500, chunk_overlap: int = 100) -> list[dict]:
    """Create text chunks with the original filename and page number."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size.")
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len, add_start_index=True)
    chunks = []
    for page in pages:
        for chunk in splitter.create_documents([page["text"]], metadatas=[page["metadata"]]):
            chunks.append({"text": chunk.page_content, "metadata": dict(chunk.metadata)})
    return chunks
