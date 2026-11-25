"""Generate context-grounded answers with the Groq chat API."""
import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

PROJECT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_DIR / ".env")
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"
SYSTEM_PROMPT = """You answer employee questions using only the supplied company document context.
1. Answer only using the supplied document context.
2. Do not use outside knowledge to fill missing information.
3. If unavailable, clearly say the information could not be found in the provided documents.
4. Do not invent policy details.
5. Keep answers simple, clear, and helpful.
6. Do not make up source names or page numbers.
"""

def generate_answer(question: str, retrieved_chunks: list[dict]) -> str:
    """Ask Groq using retrieved context; never expose the API key."""
    if not retrieved_chunks:
        return "I don't know. The information could not be found in the provided documents."
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing. Add it to the project .env file.")
    model_name = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL).strip() or DEFAULT_GROQ_MODEL
    context = "\n\n---\n\n".join(
        f"Source: {chunk['source']}, page {chunk['page']}\n{chunk['text']}" for chunk in retrieved_chunks
    )
    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Document context:\n{context}\n\nQuestion: {question}"},
            ],
            temperature=0,
        )
    except Exception as exc:
        raise RuntimeError("Groq API request failed. Check connectivity, model, and API configuration.") from exc
    answer = response.choices[0].message.content if response.choices else None
    if not answer or not answer.strip():
        raise RuntimeError("Groq returned an empty answer.")
    return answer.strip()
