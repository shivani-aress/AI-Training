import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq


CHATBOT_DIR = Path(__file__).resolve().parent
load_dotenv(CHATBOT_DIR / ".env")

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("Groq API key not found. Check your .env file.")

client = Groq(api_key=api_key)
MODEL = "openai/gpt-oss-20b"

SYSTEM_INSTRUCTION = """
You are a helpful AI chatbot.
Answer user questions in simple and clear language.
Keep answers concise unless the user requests detail.
If you do not know something, admit it.
""".strip()

conversation_history = [{"role": "system", "content": SYSTEM_INSTRUCTION}]


def main():
    print("Groq chatbot is ready. Type /clear to reset the conversation or /exit to quit.")

    while True:
        try:
            user_message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_message == "/exit":
            print("Goodbye!")
            break

        if user_message == "/clear":
            conversation_history[:] = [
                {"role": "system", "content": SYSTEM_INSTRUCTION}
            ]
            print("Conversation cleared.")
            continue

        if not user_message:
            continue

        conversation_history.append({"role": "user", "content": user_message})

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=conversation_history,
                temperature=0.7,
                max_completion_tokens=500,
            )
            assistant_response = response.choices[0].message.content
            if not assistant_response:
                raise RuntimeError("Groq returned an empty response.")

            conversation_history.append(
                {"role": "assistant", "content": assistant_response}
            )
            print(f"Bot: {assistant_response}")
        except Exception as error:
            conversation_history.pop()
            print(f"Unable to get a response from Groq: {error}")


if __name__ == "__main__":
    main()
