#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path


def sniff_dialect(input_path):
    text = input_path.read_text(encoding="utf-8-sig")
    sample = text[:4096]

    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t")
    except csv.Error:
        return csv.excel


def read_zero_rows(input_csv, label_column):
    input_path = Path(input_csv)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    dialect = sniff_dialect(input_path)
    zero_rows = []

    with input_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, dialect=dialect)

        if not reader.fieldnames:
            raise ValueError(f"Input CSV has no header row: {input_csv}")

        fieldnames = [field.strip() if field else "" for field in reader.fieldnames]
        reader.fieldnames = fieldnames

        if label_column not in fieldnames:
            raise ValueError(
                f"Input CSV must contain label column {label_column!r}. "
                f"Found columns: {', '.join(fieldnames)}"
            )

        output_fieldnames = [
            field for field in fieldnames if field and field != label_column
        ]

        if not output_fieldnames:
            raise ValueError(
                f"Input CSV has no columns left after removing {label_column!r}"
            )

        for row in reader:
            label = (row.get(label_column) or "").strip()

            if label == "0":
                zero_rows.append(
                    {
                        field: (row.get(field) or "").strip()
                        for field in output_fieldnames
                    }
                )

    return zero_rows, output_fieldnames, dialect.delimiter


def write_material_input(output_csv, rows, output_fieldnames, delimiter, overwrite):
    output_path = Path(output_csv)

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {output_csv}. "
            "Use --overwrite to replace it."
        )

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=output_fieldnames, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract rows labelled 0 from a classify_nouns*.py output CSV "
            "and write a CSV for classify_material*.py with the label column "
            "removed."
        )
    )

    parser.add_argument(
        "input_csv",
        help="Output CSV produced by a classify_nouns*.py script.",
    )

    parser.add_argument(
        "output_csv",
        help="New CSV to use as --input_csv for classify_material*.py.",
    )

    parser.add_argument(
        "--label_column",
        default="0/1",
        help="Binary label column to filter on. Defaults to 0/1.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output_csv if it already exists.",
    )

    args = parser.parse_args()

    zero_rows, output_fieldnames, delimiter = read_zero_rows(
        input_csv=args.input_csv,
        label_column=args.label_column,
    )

    write_material_input(
        output_csv=args.output_csv,
        rows=zero_rows,
        output_fieldnames=output_fieldnames,
        delimiter=delimiter,
        overwrite=args.overwrite,
    )

    print(f"Output columns: {', '.join(output_fieldnames)}")
    print(f"Rows with {args.label_column}=0: {len(zero_rows)}")
    print(f"Output written: {args.output_csv}")


if __name__ == "__main__":
    main()
