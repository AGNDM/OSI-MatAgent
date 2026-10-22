"""Fixed lexical retrieval over protocol passages only; no cases or evaluator access."""
from hashlib import sha256
import json
from pathlib import Path
import re

CORPUS = Path(__file__).resolve().parents[1] / "corpus/lab_reference.json"
QUERIES = {
    "diagnostic": "inspection balance measurement temperature sample",
    "procedure": "proposal request capability arguments revision",
    "verification": "verification fresh evidence mass criterion quality",
}
LIMIT = 2


def retrieve(role: str) -> list[dict]:
    # Queries depend only on role, so all tested models receive identical passages.
    terms = set(re.findall(r"[a-z_]+", QUERIES[role]))
    passages = json.loads(CORPUS.read_text())
    ranked = sorted(passages, key=lambda p: (
        -len(terms & set(re.findall(r"[a-z_]+", (p["title"] + " " + p["text"]).lower()))), p["id"]))
    return ranked[:LIMIT]


def manifest() -> dict:
    root = CORPUS.parent.parent
    sources = {source for passage in json.loads(CORPUS.read_text())
               for source in passage["source"].split("; ")}
    return {"backend": "fixed lexical role-query retrieval-v1", "limit": LIMIT,
        "queries": QUERIES, "corpus_sha256": sha256(CORPUS.read_bytes()).hexdigest(),
        "source_hashes": {s: sha256((root / s).read_bytes()).hexdigest() for s in sorted(sources)},
        "limitations": "Authored protocol summaries; no Materials Encoder, scenario fixtures or evaluator truth"}
