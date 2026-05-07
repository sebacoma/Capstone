"""Convert FINAL_DATA.json to JSONL format for the narrative extraction pipeline."""

import json
import re
import sys
from datetime import datetime
from pathlib import Path

# Mapping Spanish month abbreviations (for locale-independent parsing)
MONTH_MAP = {
    "ene": "Jan", "feb": "Feb", "mar": "Mar", "abr": "Apr",
    "may": "May", "jun": "Jun", "jul": "Jul", "ago": "Aug",
    "sep": "Sep", "oct": "Oct", "nov": "Nov", "dic": "Dec",
}


def parse_date(raw: str) -> str:
    """Convert heterogeneous date formats to YYYY-MM-DD ISO format.

    Handles:
        DD-MM-YYYY          (16 articles)
        D Mon YYYY           (721 articles, e.g. "18 Oct 2023")
        DD/MM/YYYY           (153 articles)
        DD/MM/YYYYhh:mm TZ  (41 articles, e.g. "22/12/202100:46 CET")
    """
    raw = raw.strip()

    # DD-MM-YYYY
    if m := re.match(r"(\d{2})-(\d{2})-(\d{4})$", raw):
        return f"{m.group(3)}-{m.group(2)}-{m.group(1)}"

    # DD/MM/YYYY with optional time suffix
    if m := re.match(r"(\d{2})/(\d{2})/(\d{4})", raw):
        return f"{m.group(3)}-{m.group(2)}-{m.group(1)}"

    # D Mon YYYY (English or Spanish month names)
    raw_norm = raw
    for es, en in MONTH_MAP.items():
        raw_norm = re.sub(rf"\b{es}\b", en, raw_norm, flags=re.IGNORECASE)

    try:
        dt = datetime.strptime(raw_norm, "%d %b %Y")
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        pass

    raise ValueError(f"Cannot parse date: {raw!r}")


def convert(input_path: str, output_path: str) -> None:
    """Convert FINAL_DATA.json to JSONL."""
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    pages = data["pages"]
    parsed = []
    errors = []

    for i, page in enumerate(pages):
        try:
            iso_date = parse_date(page["date"])
        except ValueError as e:
            errors.append((i, str(e)))
            continue

        parsed.append({
            "raw_index": i,
            "title": page["title"],
            "text": page["content"],
            "date": iso_date,
        })

    # Sort by date, then by original index for stability
    parsed.sort(key=lambda d: (d["date"], d["raw_index"]))

    # Assign sequential IDs
    for idx, doc in enumerate(parsed, start=1):
        doc["id"] = f"art_{idx:04d}"
        del doc["raw_index"]

    # Write JSONL
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for doc in parsed:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    # Summary
    dates = [d["date"] for d in parsed]
    print(f"Converted {len(parsed)} articles to {output_path}", file=sys.stderr)
    print(f"Date range: {min(dates)} to {max(dates)}", file=sys.stderr)
    if errors:
        print(f"Skipped {len(errors)} articles with unparseable dates:", file=sys.stderr)
        for idx, err in errors[:5]:
            print(f"  [{idx}] {err}", file=sys.stderr)


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    default_input = script_dir.parent.parent / "FINAL_DATA.json"
    default_output = script_dir / "corpus.jsonl"

    import argparse
    parser = argparse.ArgumentParser(description="Convert FINAL_DATA.json to JSONL")
    parser.add_argument("--input", default=str(default_input), help="Path to FINAL_DATA.json")
    parser.add_argument("--output", default=str(default_output), help="Output JSONL path")
    args = parser.parse_args()
    convert(args.input, args.output)
