"""
Extract sentences from a EMMA corpus VRT file where there exists a noun pair (PosValue == 'S') such that:
- noun1 is in elative (ela)
- noun2 is in nominative (nom)
- noun2 occurs AFTER noun1 in token order
- noun2 is the dependency head (parent) of noun1 (i.e., head(noun1) == id(noun2))

Output: elative_modifiers.csv (semicolon-separated) with rows:
noun1 noun2; sentence
"""

import csv
import re
from typing import Dict, List, Optional, Tuple

INPUT_FILE = "vrt-with-meta-corpus-02-06-25_ordered.vrt"
OUTPUT_FILE = "elative_modifiers_EMMA.csv"

# Column order in the header
HEADER = [
    "word", "word_id", "word_tokens", "animacy", "aspect", "case", "definiteness", "degree",
    "gender", "mood", "negative", "numType", "number", "person", "possessive", "pronType",
    "reflex", "tense", "transitivity", "value", "verbForm", "voice", "PosValue", "coarseValue",
    "Lemma", "correction_status", "annotator_code", "error_correction", "error_tag",
    "DependencyType", "flavor", "DependencyHead", "error_type",
]

ELATIVE = {"ela"}
NOMINATIVE = {"nom"}


def parse_case(row: Dict[str, str]) -> Optional[str]:
    """
    Try to get morphological case from either:
    - dedicated 'case' column (e.g. Nom, Gen, Ela)
    - 'value' feature bundle (e.g. Case=Nom|Number=Sing)
    Returns lowercase case string or None.
    """
    c = (row.get("case") or "").strip()
    if c and c != "_":
        return c.lower()

    feats = (row.get("value") or "").strip()
    if feats and feats != "_" and "Case=" in feats:
        for part in feats.split("|"):
            part = part.strip()
            if part.startswith("Case="):
                return part.split("=", 1)[1].strip().lower()

    return None


def detokenize(words: List[str]) -> str:
    """
    Simple detokenizer:
    - joins by spaces, then removes spaces before punctuation
    - removes spaces after opening quotes/brackets
    """
    text = " ".join(words)

    # Remove space before common punctuation
    text = re.sub(r"\s+([.,!?;:%\)\]\}…])", r"\1", text)

    # Remove space after opening brackets/quotes
    text = re.sub(r"([\(\[\{«„“])\s+", r"\1", text)

    # Handle quotes
    text = re.sub(r"\s+([»”’'])", r"\1", text)

    return text


def parse_token_line(line: str) -> Optional[Dict[str, str]]:
    parts = line.rstrip("\n").split("\t")
    if len(parts) < len(HEADER):
        return None
    # If there are extra columns, ignore them safely
    parts = parts[: len(HEADER)]
    return dict(zip(HEADER, parts))


def extract_matches_from_sentence(tokens: List[Dict[str, object]]) -> List[Tuple[str, str]]:
    """
    Returns list of (expression, sentence_text) for all matches in this sentence.
    """
    # Build sentence text from surface forms
    words = [t["word"] for t in tokens]
    sentence_text = detokenize(words)

    # Collect noun tokens
    nouns = [t for t in tokens if t["pos"] == "S" and t["case"] is not None]
    if len(nouns) < 2:
        return []

    matches: List[Tuple[str, str]] = []
    for n1 in nouns:
        if n1["case"] not in ELATIVE:
            continue
        head = n1["head"]
        if head is None:
            continue

        for n2 in nouns:
            if n2["case"] not in NOMINATIVE:
                continue
            if n2["id"] <= n1["id"]:
                continue

            # Dependency condition: noun2 is parent(head) of noun1
            if head == n2["id"]:
                expr = f"{n1['word']} {n2['word']}"
                matches.append((expr, sentence_text))

    return matches


def main() -> None:
    seen = set()  # de-duplicate by (expr, sentence_text)
    total_matches = 0
    total_sentences_with_match = 0

    in_sentence = False
    sentence_tokens: List[Dict[str, object]] = []

    with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
         open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as fout:

        writer = csv.writer(fout, delimiter=";", quoting=csv.QUOTE_MINIMAL)

        for line in fin:
            line = line.rstrip("\n")

            if line.startswith("<sentence "):
                in_sentence = True
                sentence_tokens = []
                continue

            if line.startswith("</sentence"):
                if in_sentence and sentence_tokens:
                    matches = extract_matches_from_sentence(sentence_tokens)
                    if matches:
                        total_sentences_with_match += 1
                        for expr, sent in matches:
                            key = (expr, sent)
                            if key in seen:
                                continue
                            seen.add(key)
                            writer.writerow([expr, sent])
                            total_matches += 1

                in_sentence = False
                sentence_tokens = []
                continue

            if not in_sentence:
                continue

            # Skip other XML tags inside a sentence (safety feature)
            if not line or line.startswith("<"):
                continue

            row = parse_token_line(line)
            if row is None:
                continue

            try:
                word = row["word"]
                wid = int(row["word_id"])
                pos = row["PosValue"]
                c = parse_case(row)

                head_raw = row.get("DependencyHead", "_")
                head = None if head_raw in (None, "", "_") else int(head_raw)

                # Normalize "root encoded as self-head"
                dep = row.get("DependencyType", "_")
                if head is not None and (dep == "root" or head == wid):
                    head = 0

                sentence_tokens.append({
                    "word": word,
                    "id": wid,
                    "pos": pos,
                    "case": c,
                    "head": head if head != 0 else None,
                })
            except Exception:
                # If a malformed line appears, skip it safely
                continue

    print(f"Done. Wrote: {OUTPUT_FILE}")
    print(f"Unique matches: {total_matches}")
    print(f"Sentences with ≥1 match: {total_sentences_with_match}")


if __name__ == "__main__":
    main()
