"""Load the schema text supplied to the language model."""

from pathlib import Path

SCHEMA_PATH = Path(__file__).resolve().parent / "database" / "schema.sql"


def load_schema():
    """Return the predefined schema as text; it is not executed."""
    try:
        schema = SCHEMA_PATH.read_text(encoding="utf-8").strip()
    except OSError as error:
        raise RuntimeError(f"Could not read schema file at {SCHEMA_PATH}") from error
    if not schema:
        raise RuntimeError("The database/schema.sql file is empty.")
    return schema
