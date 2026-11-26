# Internal Document Q&A Chatbot

## Objective

This learning project builds a terminal chatbot that answers employee questions from internal company policy PDFs. Answers use retrieved context, cite source filenames and page numbers, and abstain when the documents do not support an answer.

## What is RAG?

Retrieval-augmented generation (RAG) searches documents for passages relevant to a question, then provides those passages to a language model as context. This grounds answers in company documents and allows citations. The model is called only after retrieval; it is instructed not to fill gaps with outside knowledge.

## Technologies

- Python: application code.
- PyMuPDF (fitz): page-by-page PDF text extraction.
- LangChain RecursiveCharacterTextSplitter: text splitting only; no LangChain RAG chain is used.
- Sentence Transformers all-MiniLM-L6-v2: local embeddings for both document chunks and questions; no paid embedding API.
- ChromaDB: persistent local vector storage with cosine distance.
- Groq API: context-grounded answer generation.
- python-dotenv: load configuration from .env.

## Project structure

- documents/: source policy PDFs.
- src/document_loader.py: extract non-empty pages and filename/page metadata.
- src/text_chunker.py: split each page and preserve its metadata.
- src/embeddings.py: load the embedding model and encode chunks/questions.
- src/vector_store.py: persistent collection, stable IDs, upserts, and count.
- src/retriever.py: top-k cosine-distance search and threshold filtering.
- src/llm.py: Groq call with grounding rules and error handling.
- src/chatbot.py: terminal conversation loop and Q&A workflow.
- src/ingest.py: separate PDF ingestion and optional rebuild command.
- evaluation/test_questions.json: answerable and unanswerable examples.
- evaluation/evaluate.py: run examples for manual review.
- chroma_db/: persistent local vector database.
- .env and .env.example: local settings and safe template.
- requirements.txt: Python dependencies.

## Setup

Run these commands from the Rag-Project directory in Windows PowerShell:

    py -3.13 -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install -r requirements.txt

If the project virtual environment already exists, activate it and install any updated requirements:

    .\.venv\Scripts\Activate.ps1
    python -m pip install -r requirements.txt

On macOS/Linux, use python3.13 -m venv .venv, activate with source .venv/bin/activate, then run python -m pip install -r requirements.txt.

## Configure Groq

If .env does not already exist, copy the example file. If it already exists, edit it in place so existing settings are preserved:

    if (-not (Test-Path .env)) { Copy-Item .env.example .env }

Set these values in .env:

    GROQ_API_KEY=your_groq_api_key
    GROQ_MODEL=openai/gpt-oss-20b

GROQ_MODEL defaults to openai/gpt-oss-20b if omitted. Keep .env private and do not commit it.

## Add documents and ingest

Put policy PDFs such as leave_policy.pdf, work_from_home_policy.pdf, and it_support_policy.pdf in documents/. Ingestion is separate from chatbot startup:

    python -m src.ingest

This extracts non-empty pages, makes chunks, creates local embeddings, and stores text, vectors, and metadata. Stable chunk IDs and upserts prevent duplicate entries when ingestion repeats. If there are no PDFs with text, ingestion reports this without loading the embedding model.

To remove and recreate the collection from all PDFs currently in documents/:

    python -m src.ingest --rebuild

Use --rebuild after changing PDFs, chunking settings, or the embedding model. Rebuild also removes chunks belonging to files that have since been deleted or changed.

## Run the chatbot

    python -m src.chatbot

Ask questions in the terminal; type exit or quit to stop. Empty questions are rejected. Retrieval returns up to three chunks by default. Chunks that fail the distance threshold are dropped; if none remain, the bot returns “I don't know” without calling Groq.

## Run evaluation

    python -m evaluation.evaluate

The script uses the same retrieval and answer pipeline and displays each question, expected answerability, generated answer, and sources. It is for manual review; it does not automatically score answer correctness.

## Tuning retrieval

- Chunk size and overlap: change the split_pages call in src/ingest.py or its defaults in src/text_chunker.py; current defaults are 500 and 100 characters.
- Top-k: change DEFAULT_TOP_K in src/retriever.py; current default is 3.
- Distance threshold: change DEFAULT_DISTANCE_THRESHOLD in src/retriever.py. A lower cosine distance indicates a closer match. The current 0.62 value is a starting point, not a universal cutoff. Test and adjust it with your actual documents and representative questions. Similarity distance is a heuristic and cannot by itself determine whether a passage contains a complete answer.

For broad questions that explicitly mention work from home, the retriever adds a small policy hint to the search text and limits results to work_from_home_policy.pdf. The original wording is still sent to Groq. Specific WFH questions keep their wording but use the matching policy source.

Rebuild the database after changing chunking or embedding settings. When new PDFs are added, put them in documents/ and rerun ingestion. Use --rebuild if existing PDFs changed or were removed.




