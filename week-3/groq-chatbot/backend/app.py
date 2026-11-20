
import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq


# ==========================================
# 1. LOAD ENVIRONMENT VARIABLES
# ==========================================

# Locate the .env file in the project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(PROJECT_ROOT / ".env")

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("Groq API key not found. Check your .env file.")


# ==========================================
# 2. INITIALIZE FLASK AND GROQ
# ==========================================

app = Flask(__name__)

CORS(app)

client = Groq(api_key=api_key)

MODEL = "openai/gpt-oss-20b"


# ==========================================
# 3. SYSTEM INSTRUCTIONS
# ==========================================

SYSTEM_INSTRUCTION = """
You are a helpful AI chatbot.
Answer user questions in simple and clear language.
Keep answers concise unless the user requests detail.
If you do not know something, admit it.
"""


# ==========================================
# 4. CONVERSATION HISTORY
# ==========================================

conversation_history = [
    {
        "role": "system",
        "content": SYSTEM_INSTRUCTION
    }
]


# ==========================================
# 5. HOME ENDPOINT
# ==========================================

@app.route("/", methods=["GET"])
def home():

    return "Groq Chatbot Backend is running!"


# ==========================================
# 6. HEALTH CHECK ENDPOINT
# ==========================================

@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "ok",
        "model": MODEL
    })


# ==========================================
# 7. CHAT ENDPOINT
# ==========================================

@app.route("/chat", methods=["POST"])
def chat():

    global conversation_history

    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "Invalid JSON request."
        }), 400

    user_message = data.get("message")

    if not isinstance(user_message, str) or not user_message.strip():
        return jsonify({
            "error": "Please enter a valid message."
        }), 400

    user_message = user_message.strip()

    # Add user message to conversation history
    conversation_history.append({
        "role": "user",
        "content": user_message
    })

    try:

        # Send conversation to Groq
        response = client.chat.completions.create(
            model=MODEL,
            messages=conversation_history,
            temperature=0.7,
            max_completion_tokens=500
        )

        # Extract assistant response
        assistant_response = response.choices[0].message.content

        if not assistant_response:
            raise RuntimeError("Groq returned an empty response.")

        # Save assistant response for future context
        conversation_history.append({
            "role": "assistant",
            "content": assistant_response
        })

        # Return response to React
        return jsonify({
            "response": assistant_response
        })

    except Exception:

        # Remove failed user message from history
        if conversation_history[-1]["role"] == "user":
            conversation_history.pop()

        app.logger.exception("Error while contacting Groq")

        return jsonify({
            "error": "Unable to get a response from Groq. Please try again."
        }), 500


# ==========================================
# 8. CLEAR CHAT ENDPOINT
# ==========================================

@app.route("/clear", methods=["POST"])
def clear_chat():

    global conversation_history

    # Reset conversation history
    conversation_history = [
        {
            "role": "system",
            "content": SYSTEM_INSTRUCTION
        }
    ]

    return jsonify({
        "message": "Conversation cleared successfully."
    })


# ==========================================
# 9. RUN FLASK APPLICATION
# ==========================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
