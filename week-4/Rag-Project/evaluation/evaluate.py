"""Run test questions through the same pipeline for manual review."""
import json
import sys
from pathlib import Path
from src.chatbot import answer_question
from src.embeddings import load_embedding_model
from src.vector_store import get_collection

PROJECT_DIR = Path(__file__).resolve().parents[1]
QUESTIONS_PATH = PROJECT_DIR / "evaluation" / "test_questions.json"

def load_questions(path: Path = QUESTIONS_PATH) -> list[dict]:
    """Load questions and check the required fields."""
    with path.open("r", encoding="utf-8-sig") as handle:
        questions = json.load(handle)
    for item in questions:
        if not {"question", "expected_answer", "answerable"} <= item.keys():
            raise ValueError("Each question needs question, expected_answer, and answerable.")
    return questions

def main() -> None:
    try:
        questions = load_questions()
        collection = get_collection()
        if collection.count() == 0:
            raise RuntimeError("ChromaDB is empty. Run python -m src.ingest first.")
        model = load_embedding_model()
        for index, item in enumerate(questions, start=1):
            answer, sources = answer_question(item["question"], collection, model)
            print(f"\n[{index}/{len(questions)}] {item['question']}")
            print(f"Expected answerable: {item['answerable']}")
            print(f"Expected behavior: {item['expected_answer']}")
            print(f"Generated answer: {answer}")
            if sources:
                print("Retrieved sources:")
                for source, page in sorted({(m["source"], m["page"]) for m in sources}):
                    print(f"- {source}, page {page}")
            else:
                print("Retrieved sources: none passed the distance threshold")
        print("\nReview answers and citations manually; correctness is not automatically scored.")
    except Exception as exc:
        print(f"Evaluation could not run: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

if __name__ == "__main__":
    main()
