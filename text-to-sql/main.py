"""Terminal entry point for the Text-to-SQL generator."""

import os

from dotenv import load_dotenv

from llm import generate_response
from prompts import build_messages
from schema import load_schema


def main():
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        print("Missing GROQ_API_KEY. Add your Groq API key to the .env file.")
        return

    schema = load_schema()
    while True:
        question = input("\nAsk a business question (or type 'exit' to quit): ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            print("Please enter a question.")
            continue

        try:
            answer = generate_response(build_messages(question, schema))
        except (OSError, RuntimeError, ValueError) as error:
            print(f"Could not generate SQL: {error}")
            continue

        print("\nGenerated SQL and explanation\n")
        print(answer)


if __name__ == "__main__":
    main()
