import re, unicodedata

# Keep apostrophes inside tokens (e.g., n'go, ng'e) to match NeMo-style tokenization.
_punct = re.compile(r"[^\w\s'\u2019\u00C0-\u024F\u1E00-\u1EFF]")
_ws = re.compile(r"\s+")

def normalize(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = unicodedata.normalize("NFC", text)
    t = t.lower()
    t = _punct.sub(" ", t)
    t = _ws.sub(" ", t).strip()
    return t
