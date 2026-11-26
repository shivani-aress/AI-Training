"""Prompt templates for generating and explaining SQL."""

SYSTEM_PROMPT = """You translate business questions into SQL and explain the result.
Use only tables and columns present in the supplied schema. Never invent schema
objects. Do not execute SQL or claim that you did. If the question cannot be
answered from the schema, say so and explain which information is missing.
Generate a read-only SELECT query only. Make reasonable assumptions explicit.
Return exactly these two sections, with no Markdown fences:
SQL:
<the SQL query, or a brief statement that it cannot be generated>
EXPLANATION:
<a plain-language explanation of what the query does and its main clauses>
"""


def build_messages(question, schema):
    """Build the system and user messages sent to Groq."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Database schema:\n{schema}\n\nBusiness question:\n{question}\n\nGenerate a query using only this schema, then explain it.",
        },
    ]
