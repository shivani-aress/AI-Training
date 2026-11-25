"""Read PDF documents one page at a time and retain source metadata."""
from pathlib import Path
import pymupdf

PROJECT_DIR = Path(__file__).resolve().parents[1]
DOCUMENTS_DIR = PROJECT_DIR / "documents"

def load_pdf_pages(documents_dir: Path = DOCUMENTS_DIR) -> list[dict]:
    """Return non-empty PDF pages with source filename and 1-based page number."""
    if not documents_dir.exists():
        raise FileNotFoundError(f"Documents folder does not exist: {documents_dir}")
    if not documents_dir.is_dir():
        raise NotADirectoryError(f"Documents path is not a folder: {documents_dir}")
    pdf_paths = sorted(documents_dir.glob("*.pdf"))
    if not pdf_paths:
        return []
    pages = []
    for pdf_path in pdf_paths:
        try:
            with pymupdf.open(pdf_path) as pdf:
                for page_index, page in enumerate(pdf):
                    text = page.get_text().strip()
                    if text:
                        pages.append({
                            "text": text,
                            "metadata": {"source": pdf_path.name, "page": page_index + 1},
                        })
        except (pymupdf.FileDataError, OSError, RuntimeError) as exc:
            raise ValueError(f"Could not extract text from {pdf_path.name}: {exc}") from exc
    return pages
