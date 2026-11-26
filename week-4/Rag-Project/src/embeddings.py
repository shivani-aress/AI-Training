"""Load one Sentence Transformers model for both documents and questions."""
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

def load_embedding_model(model_name: str = EMBEDDING_MODEL_NAME) -> SentenceTransformer:
    """Load the local sentence embedding model (downloads on first use)."""
    return SentenceTransformer(model_name)

def embed_texts(model: SentenceTransformer, texts: list[str]) -> list[list[float]]:
    """Generate normalized embeddings for document chunks."""
    if not texts:
        return []
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return vectors.tolist()

def embed_query(model: SentenceTransformer, question: str) -> list[float]:
    """Embed one question using the same model."""
    if not question.strip():
        raise ValueError("The question must not be empty.")
    return model.encode(question, normalize_embeddings=True, show_progress_bar=False).tolist()
