import re


def clean_text(text: str) -> str:

    # Replace multiple spaces and tabs
    text = re.sub(r"[ \t]+", " ", text)

    # Replace excessive newlines
    text = re.sub(r"\n+", "\n", text)

    # Remove spaces at the beginning and end
    text = text.strip()

    return text
