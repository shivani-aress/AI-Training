# Company Documents RAG Chatbot

A beginner-friendly Python project that will answer questions using company documents such as HR, salary, and benefits policies. Sample PDFs can be added to `documents/` later.

## Technology stack

- Python for the application logic
- PyPDF (`pypdf`) to read PDF text
- SentenceTransformers with `all-MiniLM-L6-v2` to create text embeddings
- ChromaDB to store and retrieve document chunks
- Groq API to generate answers from retrieved context
- `python-dotenv` to load environment variables

## Files and folders

- `documents/`: Place policy PDF files here when they are ready.
- `chroma_db/`: Persistent storage location for the vector database.
- `ingest.py`: Planned entry point for reading PDFs, chunking text, creating embeddings, and storing data.
- `chatbot.py`: Planned entry point for asking questions and generating answers with retrieved context.
- `.env`: Local environment variables; add your own `GROQ_API_KEY` when ready.
- `.env.example`: Example environment-variable setting.
- `requirements.txt`: Python dependencies for the project.
- `README.md`: Project overview and file guide.

The RAG pipeline has not been implemented yet.
