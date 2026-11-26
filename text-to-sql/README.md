# Text-to-SQL

A Python terminal application that turns a plain-English business question into a SQL query and a simple explanation using the Groq API and a predefined schema.

## Setup

1. Use Python 3.9 or newer.
2. From this project directory, install dependencies with `pip install -r requirements.txt`.
3. Add your Groq API key to `.env` as `GROQ_API_KEY=...`. The default model is configured by `GROQ_MODEL` and can be changed there.
4. Start the app with `python main.py`.

## Example

Ask: `Show the top 5 products with the highest sales.`

The sample schema lets the model build a query that joins `products` and `order_items`, sums quantity times unit price, groups by product, and sorts by total sales. The response includes the SQL and a plain-language explanation.

## Project files

- `main.py`: reads the question and coordinates schema loading, prompting, and output.
- `llm.py`: calls Groq chat completions and returns the generated text.
- `schema.py`: reads `database/schema.sql` as prompt context.
- `prompts.py`: defines the generation rules and message format.
- `database/schema.sql`: provides the sample retail tables and columns.

This is a generation-and-explanation tool only. It does not connect to a database or execute, validate, or modify SQL. Review generated SQL before using it elsewhere.
