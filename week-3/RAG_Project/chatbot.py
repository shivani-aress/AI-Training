"""Search company documents and answer questions using their retrieved context."""

import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-20b"
DATABASE_PATH = "./chroma_db"
COLLECTION_NAME = "company_documents"
# The API key is stored in week-3/.env, one folder above this project.
ENV_FILE = Path(__file__).parent.parent / ".env"

SYSTEM_INSTRUCTIONS = """Answer questions using only the provided company-document context.
Do not invent company policies or information.
If the context does not contain the answer, clearly say that the information is not available in the provided documents.
Keep answers simple and concise."""


def main():
    """Ask questions, retrieve relevant chunks, and get grounded answers."""
    # Load the key from the existing week-3 .env file; never put it in the code.
    load_dotenv(dotenv_path=ENV_FILE)
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print(f"GROQ_API_KEY was not found. Add it to this file: {ENV_FILE}")
        return

    # Connect to the database created by ingest.py and open its collection.
    client = chromadb.PersistentClient(path=DATABASE_PATH)
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception as error:
        print(f"Could not open the '{COLLECTION_NAME}' collection: {error}")
        print("Run ingest.py first to create and fill the collection.")
        return

    if collection.count() == 0:
        print("The company_documents collection is empty. Run ingest.py first to add documents.")
        return

    # Load the same model used by ingest.py; it is used here only for questions.
    model = SentenceTransformer(MODEL_NAME)
    groq_client = Groq(api_key=api_key)

    print("Ask questions about the company documents. Type 'exit' to quit.")

    while True:
        question = input("\nYour question: ").strip()

        if question.lower() == "exit":
            print("Goodbye!")
            break

        if not question:
            print("Please enter a question, or type 'exit' to quit.")
            continue

        # Embed the question and retrieve the three closest stored chunks.
        question_embedding = model.encode(question).tolist()
        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=min(3, collection.count()),
            include=["documents", "metadatas"],
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]

        if not documents:
            print("No matching document chunks were found, so I cannot answer from the documents.")
            continue

        # Keep retrieved text for Groq, but do not print the chunks to the terminal.
        context_sections = []

        for document, metadata in zip(documents, metadatas):
            metadata = metadata or {}
            source = metadata.get("source", "Unknown")
            page = metadata.get("page", "Unknown")

            # Add source labels to the context so they stay associated with each excerpt.
            context_sections.append(
                f"Source: {source} | Page: {page}\n{document}"
            )

        context = "\n\n".join(context_sections)
        user_message = f"Company-document context:\n{context}\n\nQuestion: {question}"

        # Ask Groq to answer using only the retrieved context and the user question.
        try:
            response = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                    {"role": "user", "content": user_message},
                ],
            )
            answer = response.choices[0].message.content
            print(f"\nYou: {question}")
            print(f"Assistant: {answer or 'The model returned an empty answer.'}")
        except Exception as error:
            # Keep an API failure from ending the question loop.
            print(f"The Groq API request failed: {error}")


if __name__ == "__main__":
    main()
