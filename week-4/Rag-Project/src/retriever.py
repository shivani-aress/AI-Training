"""Retrieve relevant chunks separately from LLM generation."""
from .embeddings import embed_query

DEFAULT_TOP_K = 3
DEFAULT_DISTANCE_THRESHOLD = 0.62

# These terms identify broad questions about the work-from-home policy.
WORK_FROM_HOME_TERMS = (
    "work from home",
    "work-from-home",
    "wfh",
    "work remotely",
    "remote work",
)
WORK_FROM_HOME_SOURCE = "work_from_home_policy.pdf"
WORK_FROM_HOME_DETAIL_TERMS = (
    "day",
    "days",
    "advance",
    "notice",
    "request",
    "approval",
    "manager",
    "hour",
    "schedule",
    "internet",
    "travel",
    "country",
    "city",
    "vpn",
    "security",
    "eligible",
)


def _is_work_from_home_question(question: str) -> bool:
    """Check whether the question clearly names the WFH topic."""
    return any(term in question.casefold() for term in WORK_FROM_HOME_TERMS)


def prepare_search_query(question: str) -> str:
    """Add policy terminology to broad WFH queries to improve matching."""
    normalized = question.casefold()
    is_work_from_home_question = _is_work_from_home_question(question)
    asks_about_a_specific_detail = any(
        term in normalized for term in WORK_FROM_HOME_DETAIL_TERMS
    )

    if is_work_from_home_question and not asks_about_a_specific_detail:
        return f"{question} work from home policy eligibility approval arrangement"
    return question


def retrieve(
    collection,
    model,
    question: str,
    top_k: int = DEFAULT_TOP_K,
    distance_threshold: float | None = DEFAULT_DISTANCE_THRESHOLD,
) -> list[dict]:
    """Return text, source, page, and cosine distance, best match first."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")
    if not question.strip():
        raise ValueError("The question must not be empty.")

    count = collection.count()
    if count == 0:
        return []

    # The expanded text is only for search; the original question goes to the LLM.
    search_query = prepare_search_query(question)
    query_embedding = embed_query(model, search_query)

    # This explicit topic match avoids unrelated policies crowding WFH results.
    query_options = {}
    if _is_work_from_home_question(question):
        query_options["where"] = {"source": WORK_FROM_HOME_SOURCE}

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"],
        **query_options,
    )

    matches = []
    for text, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        if distance_threshold is not None and distance > distance_threshold:
            continue
        matches.append(
            {
                "text": text,
                "source": metadata["source"],
                "page": metadata["page"],
                "distance": float(distance),
            }
        )
    return matches
