
"""Read-only loading and safe access helpers for audit JSON."""

import json
from pathlib import Path
from typing import Any


def load_audit_json(source: Any) -> dict:
    """Load an audit record from a path, bytes, string, or uploaded file."""

    # Handle empty string before treating strings as file paths.
    if isinstance(source, str) and not source.strip():
        raise ValueError("The audit JSON file is empty.")

    if isinstance(source, (str, Path)):
        text = Path(source).read_text(encoding="utf-8")
    elif isinstance(source, bytes):
        text = source.decode("utf-8-sig")
    elif hasattr(source, "read"):
        content = source.read()
        if isinstance(content, bytes):
            text = content.decode("utf-8-sig")
        else:
            text = content
    else:
        raise TypeError(
            "Source must be a file path, bytes, string, or file-like object."
        )

    if not text or not text.strip():
        raise ValueError("The audit JSON file is empty.")

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid audit JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(
            "The audit JSON must contain a JSON object at the top level."
        )

    return data


def get_nested(data: dict, *keys: str, default: Any = None) -> Any:
    """Safely retrieve a value from nested dictionaries."""

    current = data

    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]

    return current


def safe_list(value: Any) -> list:
    """Return a list only when the supplied value is actually a list."""

    return value if isinstance(value, list) else []


def safe_dict(value: Any) -> dict:
    """Return a dictionary only when the supplied value is actually a dictionary."""

    return value if isinstance(value, dict) else {}


def display_value(value: Any, fallback: str = "Not available") -> str:
    """Convert missing or empty values into a readable display string."""

    if value is None or value == "":
        return fallback

    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, indent=2)

    return str(value)
