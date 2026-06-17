#!/usr/bin/env python3

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from urllib import request, error


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_INPUT_CSV = Path(__file__).resolve().parents[1] / "test_data" / "instance_lemma_comp1_form_comp2_lemma.csv"

SYSTEM_PROMPT = """Sa oled leksikograaf, kes töötab eestikeelsete nimisõnafraaside klassifitseerimisega.

Sulle antakse täpselt üks fraas, selle esimene sõna ja teise sõna lemma.
OTSUSTA, kas selles fraasis väljendab esimene sõna teise sõna ELUKUTSET, RAHVUST või ROLLI.

Vasta ainult ühe märgiga:
1 = JAH
0 = EI

Ära lisa selgitust.
"""


def load_env_file(env_path):
    env_path = Path(env_path)

    if not env_path.exists():
        return

    with env_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            if key and key not in os.environ:
                os.environ[key] = value


def read_words(input_csv):
    input_path = Path(input_csv)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    text = input_path.read_text(encoding="utf-8-sig")
    sample = text[:2048]

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
    except csv.Error:
        class SemicolonDialect(csv.excel):
            delimiter = ";"

        dialect = SemicolonDialect

    phrase_rows = []

    with input_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, dialect=dialect)
        input_headers = reader.fieldnames or []

        if not input_headers:
            raise ValueError(f"Input CSV must contain a header row: {input_csv}")

        first_header = input_headers[0]

        if not first_header or not first_header.strip():
            raise ValueError(f"Input CSV first header cannot be empty: {input_csv}")

        has_detailed_columns = {"instance_lemma", "comp1_form", "comp2_lemma"}.issubset(input_headers)

        for row in reader:
            input_values = [(row.get(header) or "").strip() for header in input_headers]

            if not input_values or not input_values[0]:
                continue

            phrase_row = {
                "input_values": input_values,
                "target_text": input_values[0],
            }

            if has_detailed_columns:
                instance_lemma = (row.get("instance_lemma") or "").strip()
                comp1_form = (row.get("comp1_form") or "").strip()
                comp2_lemma = (row.get("comp2_lemma") or "").strip()

                if not (instance_lemma and comp1_form and comp2_lemma):
                    continue

                phrase_row.update(
                    {
                        "instance_lemma": instance_lemma,
                        "comp1_form": comp1_form,
                        "comp2_lemma": comp2_lemma,
                    }
                )

            phrase_rows.append(phrase_row)

    return input_headers, phrase_rows


def read_existing_answers(output_csv, input_headers):
    output_path = Path(output_csv)

    if not output_path.exists():
        return set()

    done = Counter()

    with output_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        expected_headers = input_headers + ["0/1"]

        if reader.fieldnames != expected_headers:
            raise ValueError(
                f"Existing output header {reader.fieldnames} does not match expected {expected_headers}. "
                "Use a new output file or rerun with --overwrite."
            )

        for row in reader:
            row_key = tuple((row.get(header) or "").strip() for header in input_headers)
            answer = (row.get("0/1") or "").strip()

            if all(row_key) and answer in {"0", "1"}:
                done[row_key] += 1

    return done


def init_output_file(output_csv, overwrite, input_headers):
    output_path = Path(output_csv)

    if overwrite and output_path.exists():
        output_path.unlink()

    if not output_path.exists():
        with output_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(input_headers + ["0/1"])


def normalize_answer(answer_text):
    if answer_text is None:
        return "0"

    text = str(answer_text).strip()

    if text == "":
        return "0"

    if text.lower() in {"none", "null", "nan"}:
        return "0"

    if text in {"0", "1"}:
        return text

    if text.startswith("1"):
        return "1"

    if text.startswith("0"):
        return "0"

    upper_text = text.upper()

    if "JAH" in upper_text and "EI" not in upper_text:
        return "1"

    if "EI" in upper_text and "JAH" not in upper_text:
        return "0"

    digits = re.findall(r"(?<!\d)[01](?!\d)", text)

    if len(digits) == 1:
        return digits[0]

    raise ValueError(f"Could not normalize model answer: {answer_text!r}")

def extract_content(response_json):
    if "error" in response_json:
        raise RuntimeError(response_json["error"])

    choices = response_json.get("choices", [])

    if not choices:
        raise RuntimeError(f"No choices in response: {response_json}")

    choice = choices[0]
    finish_reason = choice.get("finish_reason")
    message = choice.get("message", {})
    content = message.get("content")

    if content is None:
        raise RuntimeError(
            f"Empty model content. finish_reason={finish_reason}, response={response_json}"
        )

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []

        for item in content:
            if isinstance(item, dict):
                parts.append(item.get("text", ""))
            else:
                parts.append(str(item))

        return "".join(parts)

    return str(content)


def call_openrouter(api_key, model_name, phrase_row, timeout, max_retries):
    if {"instance_lemma", "comp1_form", "comp2_lemma"}.issubset(phrase_row):
        instance_lemma = phrase_row["instance_lemma"]
        comp1_form = phrase_row["comp1_form"]
        comp2_lemma = phrase_row["comp2_lemma"]
        user_prompt = f"""Analüüsitav fraas: "{instance_lemma}"
Esimene sõnavorm: "{comp1_form}"
Põhisõna lemma: "{comp2_lemma}"

OTSUSTA, kas fraasis "{instance_lemma}" väljendab sõna "{comp1_form}" sõna "{comp2_lemma}" ELUKUTSET, RAHVUST või ROLLI.

Vasta ainult ühe märgiga: 1 või 0.
"""
    else:
        instance_lemma = phrase_row["target_text"]
        user_prompt = f"""Analüüsitav fraas: "{instance_lemma}"

OTSUSTA, kas fraasi "{instance_lemma}" esimene komponent väljendab põhisõna ELUKUTSET, RAHVUST või ROLLI.

Vasta ainult ühe märgiga: 1 või 0.
"""

    payload = {
    "model": model_name,
    "messages": [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ],
    "temperature": 0,
}

    if "gemma" in model_name.lower():
        payload.update({
            "max_completion_tokens": 2048,
            "reasoning": {
                "effort": "none"
            }
        })
    else:
        payload.update({
            "max_tokens": 16,
            "reasoning": {
                "effort": "none"
            },
            "stop": ["\n"]
        })

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "Estonian noun classifier"
    }

    data = json.dumps(payload).encode("utf-8")

    for attempt in range(1, max_retries + 1):
        req = request.Request(
            OPENROUTER_URL,
            data=data,
            headers=headers,
            method="POST"
        )

        try:
            with request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8")
                response_json = json.loads(body)
                return extract_content(response_json)

        except error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")

            if e.code in {429, 500, 502, 503, 504} and attempt < max_retries:
                wait_seconds = min(2 ** attempt, 30)
                print(
                    f"HTTP {e.code}. Retry {attempt}/{max_retries} after {wait_seconds}s...",
                    file=sys.stderr
                )
                time.sleep(wait_seconds)
                continue

            raise RuntimeError(f"OpenRouter HTTP error {e.code}: {body}")

        except error.URLError as e:
            if attempt < max_retries:
                wait_seconds = min(2 ** attempt, 30)
                print(
                    f"Network error: {e}. Retry {attempt}/{max_retries} after {wait_seconds}s...",
                    file=sys.stderr
                )
                time.sleep(wait_seconds)
                continue

            raise RuntimeError(f"Network error: {e}")

    raise RuntimeError("OpenRouter request failed after retries")


def classify_word(api_key, model_name, phrase_row, timeout, max_retries):
    answer_text = call_openrouter(
        api_key=api_key,
        model_name=model_name,
        phrase_row=phrase_row,
        timeout=timeout,
        max_retries=max_retries
    )

    return normalize_answer(answer_text)


def append_answer(output_csv, phrase_row, answer):
    with open(output_csv, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(phrase_row["input_values"] + [answer])
        f.flush()


def main():
    parser = argparse.ArgumentParser(
        description="Classify Estonian nouns via OpenRouter: 1=profession/nationality/role, 0=not."
    )

    parser.add_argument(
        "model_name",
        help="OpenRouter model id, e.g. google/gemini-2.5-flash-lite"
    )

    parser.add_argument(
        "output_csv",
        nargs="?",
        default="classified_nouns_2_1.csv",
        help="Output CSV file, e.g. classified_nouns_2_1.csv"
    )

    parser.add_argument(
        "--input_csv",
        default=DEFAULT_INPUT_CSV,
        help="Input CSV file with a header row and one phrase per row"
    )

    parser.add_argument(
        "--env_file",
        default=".env",
        help="Path to .env file containing OPENROUTER_API_KEY"
    )

    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Optional sleep between API calls in seconds"
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="HTTP timeout in seconds"
    )

    parser.add_argument(
        "--max_retries",
        type=int,
        default=4,
        help="Maximum retry attempts for failed API calls"
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output CSV instead of resuming"
    )

    args = parser.parse_args()

    load_env_file(args.env_file)

    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY not found. Put it in .env as OPENROUTER_API_KEY=sk-or-..."
        )

    input_headers, phrase_rows = read_words(args.input_csv)

    if not phrase_rows:
        raise RuntimeError(f"No phrases found in {args.input_csv}")

    init_output_file(args.output_csv, args.overwrite, input_headers)
    done_words = read_existing_answers(args.output_csv, input_headers)

    todo_rows = []
    seen_rows = Counter()

    for phrase_row in phrase_rows:
        row_key = tuple(phrase_row["input_values"])

        seen_rows[row_key] += 1

        if seen_rows[row_key] > done_words[row_key]:
            todo_rows.append(phrase_row)

    print(f"Input phrases: {len(phrase_rows)}")
    print(f"Already done: {sum(done_words.values())}")
    print(f"Remaining: {len(todo_rows)}")
    print(f"Model: {args.model_name}")
    print(f"Output: {args.output_csv}")

    for idx, phrase_row in enumerate(todo_rows, start=1):
        word = phrase_row["target_text"]

        try:
            answer = classify_word(
                api_key=api_key,
                model_name=args.model_name,
                phrase_row=phrase_row,
                timeout=args.timeout,
                max_retries=args.max_retries
            )

            append_answer(args.output_csv, phrase_row, answer)

            print(f"[{idx}/{len(todo_rows)}] {word} -> {answer}")

            if args.sleep > 0:
                time.sleep(args.sleep)

        except Exception as e:
            print(f"FAILED on phrase: {word}", file=sys.stderr)
            print(str(e), file=sys.stderr)
            print(
                "Progress so far has been saved. Re-run the same command to resume.",
                file=sys.stderr
            )
            sys.exit(1)

    print("Done.")


if __name__ == "__main__":
    main()
