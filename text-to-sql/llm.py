"""Groq API integration. This module generates text and never runs SQL."""

import os

from groq import Groq


def generate_response(messages):
    """Send the prompt to Groq and return the model's text response."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set.")

    try:
        completion = Groq(api_key=api_key).chat.completions.create(
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
            messages=messages,
            temperature=0.2,
        )
    except Exception as error:
        raise RuntimeError(f"Groq request failed: {error}") from error

    content = completion.choices[0].message.content
    if not content:
        raise RuntimeError("Groq returned an empty response.")
    return content.strip()
