import json
import os
import re
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from groq import Groq


# Keep paths relative to this project so the script is easy to run locally.
project_dir = Path(__file__).parent
csv_path = project_dir / "input" / "support_tickets.csv"
env_path = project_dir / ".env"
output_path = project_dir / "output" / "processed_tickets.csv"

# Choose the Groq model in one clearly defined place.
MODEL_NAME = "openai/gpt-oss-120b"
OUTPUT_COLUMNS = ["ticket_id", "message", "summary", "category", "priority"]

# Match common phone formats such as +1-202-555-0100 or (202) 555-0100.
PHONE_PATTERN = re.compile(
    r"(?<!\d)(?:\+?1[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}(?!\d)"
)


def redact_phone_numbers(text):
    """Replace phone-like numbers before any message is sent to the LLM."""
    return PHONE_PATTERN.sub("[PHONE REDACTED]", text)


try:
    # Load customer support records into a DataFrame.
    df = pd.read_csv(csv_path)

    # Avoid displaying phone numbers in the console output.
    display_df = df.copy()
    if "phone_number" in display_df.columns:
        display_df["phone_number"] = display_df["phone_number"].apply(
            lambda value: "[PHONE REDACTED]" if pd.notna(value) and str(value).strip() else ""
        )

    print("Complete dataset:")
    print(display_df)
    print(f"\nTotal number of records: {len(df)}")
    print(f"Column names: {list(df.columns)}")
    print("\nMissing values in each column:")
    print(df.isnull().sum())

    if "message" not in df.columns or "ticket_id" not in df.columns:
        print("Error: The CSV file must contain 'ticket_id' and 'message' columns.")
        raise SystemExit(1)

    # Trim extra spaces and keep only rows with a non-empty message.
    # The original CSV is not changed; preprocessing happens in memory.
    df["message"] = df["message"].astype("string").str.strip()
    valid_rows = df["message"].notna() & df["message"].ne("")
    valid_df = df.loc[valid_rows].copy()
    skipped_count = len(df) - len(valid_df)

    print(f"\nValid records: {len(valid_df)}")
    print(f"Skipped records with missing or empty messages: {skipped_count}")

    if valid_df.empty:
        print("Error: There are no valid customer messages to send.")
        raise SystemExit(1)

    # Load the API key from this project's .env file without displaying it.
    load_dotenv(dotenv_path=env_path)
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_api_key_here":
        print("Error: GROQ_API_KEY is missing or still has the placeholder value in .env.")
        raise SystemExit(1)

    # Create one client and reuse it while processing tickets sequentially.
    client = Groq(api_key=api_key)
    results = []

    for _, ticket in valid_df.iterrows():
        ticket_id = ticket["ticket_id"]

        # Send a placeholder instead of the separate phone number. Also redact
        # phone-like numbers that might appear inside the customer's message.
        phone_number = ticket.get("phone_number")
        safe_phone_number = (
            "[PHONE REDACTED]"
            if pd.notna(phone_number) and str(phone_number).strip()
            else "[NOT PROVIDED]"
        )
        safe_message = redact_phone_numbers(str(ticket["message"]))

        prompt = f"""Analyze this customer support message.

Return only a valid JSON object with exactly these keys:
- summary: summarize the issue in one short sentence
- category: a suitable support category
- priority: exactly High, Medium, or Low

Customer phone number: {safe_phone_number}
Customer message:
{safe_message}

Example format:
{{"summary": "Customer is unable to log in.", "category": "Login Issue", "priority": "High"}}"""

        try:
            # Send one ticket at a time and ask Groq to return JSON.
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            response_text = response.choices[0].message.content
            result = json.loads(response_text)

            # Make sure the result has the expected fields and priority value.
            required_fields = {"summary", "category", "priority"}
            if not isinstance(result, dict) or not required_fields.issubset(result):
                raise ValueError("The response JSON is missing one or more required fields.")
            if result["priority"] not in {"High", "Medium", "Low"}:
                raise ValueError("Priority must be High, Medium, or Low.")

            # Save the source fields with the model's three output fields.
            results.append(
                {
                    "ticket_id": ticket_id,
                    "message": ticket["message"],
                    "summary": result["summary"],
                    "category": result["category"],
                    "priority": result["priority"],
                }
            )

        except Exception as error:
            # Report this ticket's failure and continue with the next one.
            print(f"Ticket {ticket_id} failed: {error}")

    # Save successful results using the required column order.
    results_df = pd.DataFrame(results, columns=OUTPUT_COLUMNS)
    results_df.to_csv(output_path, index=False)

    print(f"\nSuccessfully processed {len(results)} of {len(valid_df)} valid tickets.")
    print(f"Results saved to: {output_path}")

except FileNotFoundError:
    print(f"Error: CSV file was not found: {csv_path}")
except pd.errors.EmptyDataError:
    print(f"Error: CSV file is empty and contains no data or headers: {csv_path}")
except Exception as error:
    print(f"Error: Could not load or preprocess the CSV file: {error}")
