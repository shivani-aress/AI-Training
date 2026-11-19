
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create Gemini client
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

# System prompt
system_prompt = """
You are a friendly AI assistant.

Instructions:
1. Answer in simple language.
2. Keep answers concise.
3. Give examples wherever useful.
4. If you don't know something, admit it.
"""

print("Welcome to Gemini Chatbot!")
print("Type 'exit' to quit.")

while True:

    # Get user input
    user_prompt = input("\nYou: ")

    if user_prompt.lower() == "exit":
        print("Goodbye!")
        break

    try:
        # Send system and user prompts to Gemini
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt
            )
        )

        # Display response
        print("\nGemini:", response.text)

    except Exception as e:
        print("Something went wrong:", e)

client.close()
