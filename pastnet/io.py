"""Small, deterministic artifact helpers."""

import hashlib
import json
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_text(value):
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json_text(value), encoding="utf-8")
    temporary.replace(path)


def write_once(path, contents):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != contents:
            raise FileExistsError(f"Different existing artifact: {path}; use another output directory")
    else:
        path.write_bytes(contents)


def frame_digest(frame):
    """Hash ordered SMILES/labels, independent of CSV newline and integer formatting."""
    rows = [[str(s), float(y)] for s, y in frame[["smiles", "target"]].itertuples(index=False, name=None)]
    payload = json.dumps(rows, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()
