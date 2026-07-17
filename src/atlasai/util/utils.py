from pathlib import Path


def load_file(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")

def upsert_file(path: str, content: str):
    if not Path(path).write_text(content):
        raise ValueError("Could't write to path")