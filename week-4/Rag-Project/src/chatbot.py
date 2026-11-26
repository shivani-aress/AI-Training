"""Terminal interface and end-to-end question answering workflow."""
from .embeddings import load_embedding_model
from .llm import generate_answer
from .retriever import DEFAULT_DISTANCE_THRESHOLD, DEFAULT_TOP_K, retrieve
from .vector_store import get_collection

FALLBACK_ANSWER = "I don't know. The information could not be found in the provided documents."

def answer_question(question: str, collection, embedding_model, top_k: int = DEFAULT_TOP_K, distance_threshold: float | None = DEFAULT_DISTANCE_THRESHOLD) -> tuple[str, list[dict]]:
    """Retrieve evidence, abstain if distant, otherwise ask Groq."""
    matches = retrieve(collection, embedding_model, question, top_k=top_k, distance_threshold=distance_threshold)
    if not matches:
        return FALLBACK_ANSWER, []
    return generate_answer(question, matches), matches

def main() -> None:
    """Run the interactive terminal loop."""
    print("Welcome to the Internal Document Q&A Chatbot.")
    try:
        collection = get_collection()
        if collection.count() == 0:
            print("The vector database is empty. Run: python -m src.ingest")
            return
        print(f"Loaded {collection.count()} document chunks.")
        print("Ask a question, or type 'exit' or 'quit' to stop.")
        model = load_embedding_model()
        while True:
            question = input("\nQuestion: ").strip()
            if question.lower() in {"exit", "quit"}:
                print("Goodbye.")
                break
            if not question:
                print("Please enter a question.")
                continue
            try:
                answer, sources = answer_question(question, collection, model)
                print(f"\nAnswer: {answer}")
                if sources:
                    print("Sources:")
                    for source, page in sorted({(m["source"], m["page"]) for m in sources}):
                        print(f"- {source}, page {page}")
            except Exception as exc:
                print(f"Could not answer this question: {exc}")
    except Exception as exc:
        print(f"Could not start the chatbot: {exc}")

if __name__ == "__main__":
    main()

