
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

system_prompt = """
You are a friendly AI assistant.
Answer in simple language.
Keep answers concise.
Give examples wherever useful.
If you don't know something, admit it.
"""

prompts = [
    "Explain Retrieval-Augmented Generation (RAG) to a beginner.",

    """Act as an AI/ML tutor.
    Explain Retrieval-Augmented Generation (RAG) to a beginner.""",

    """Explain Retrieval-Augmented Generation (RAG) in simple language.
    Avoid technical jargon and keep the explanation under 100 words.""",

    """Explain Retrieval-Augmented Generation (RAG) to a beginner.

    Structure your answer into:
    1. What is RAG?
    2. How does it work?
    3. A real-world example.

    Use simple language.""",

    """I am learning about Large Language Models and already understand
    basic concepts like tokens and embeddings.

    Explain Retrieval-Augmented Generation (RAG) as if you are teaching a beginner.

    Start with a simple definition, explain the retrieval and generation steps,
    and give a practical example of a chatbot answering questions from company documents.

    Use simple language, avoid unnecessary technical details,
    and keep the answer under 150 words."""
]

try:
    for i, prompt in enumerate(prompts, start=1):

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )

        print(f"\nPROMPT {i}:")
        print(prompt)

        print("\nRESPONSE:")
        print(response.choices[0].message.content)

        print("\n" + "=" * 60)

except Exception as e:
    print("Something went wrong:", e)

finally:
    client.close()
