"""Create or sync LangSmith datasets from local JSONL files.

Accepts two JSONL formats:
  Flat:    {"input_text": "...", "output": {...}}   (used by the new data files)
  Wrapped: {"inputs": {"input_text": "..."}, "outputs": {...}}  (legacy format)

Usage:
    python scripts/seed_datasets.py               # add missing examples
    python scripts/seed_datasets.py --sync        # delete extras, re-add all
"""

import argparse
import json
import sys
from pathlib import Path

import jsonschema
from langsmith import Client

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SCHEMA_PATH = (
    Path(__file__).resolve().parent.parent / "evaluation" / "schemas" / "entities.schema.json"
)

DATASETS = {
    "summarization-regression": "summarization_regression.jsonl",
    "extraction-regression": "extraction_regression.jsonl",
}


def _load_entity_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text())


def _normalize_row(row: dict) -> dict:
    """Return (inputs dict, outputs dict or None) from either JSONL format."""
    if "inputs" in row:
        return {"inputs": row["inputs"], "outputs": row.get("outputs")}
    # Flat format.
    inputs = {"input_text": row["input_text"]}
    outputs = row.get("output")
    return {"inputs": inputs, "outputs": outputs}


def _validate_extraction_examples(rows: list[dict], schema: dict) -> None:
    for i, row in enumerate(rows, 1):
        norm = _normalize_row(row)
        outputs = norm["outputs"]
        if outputs is None:
            print(f"  line {i}: WARNING no outputs field")
            continue

        try:
            jsonschema.validate(outputs, schema)
        except jsonschema.ValidationError as exc:
            print(f"  line {i}: FAIL schema validation: {exc.message}", file=sys.stderr)
            sys.exit(1)

        for key in ("names", "dates", "locations"):
            vals = outputs.get(key, [])
            if len(vals) != len(set(vals)):
                print(
                    f"  line {i}: FAIL duplicates in '{key}': {vals}", file=sys.stderr
                )
                sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed LangSmith datasets from JSONL files."
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="delete all existing examples and re-seed from the local files",
    )
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv

        load_dotenv(override=False)
    except ImportError:
        pass

    client = Client()
    entity_schema = _load_entity_schema()

    for ds_name, filename in DATASETS.items():
        filepath = DATA_DIR / filename
        if not filepath.exists():
            print(f"FAIL: {filepath} not found", file=sys.stderr)
            sys.exit(1)

        raw_rows = []
        for line in filepath.read_text().strip().splitlines():
            line = line.strip()
            if line:
                raw_rows.append(json.loads(line))

        print(f"loaded {len(raw_rows)} examples from {filename}")

        if ds_name == "extraction-regression":
            _validate_extraction_examples(raw_rows, entity_schema)
            print("  extraction examples validated against schema")

        try:
            ds = client.read_dataset(dataset_name=ds_name)
            print(f"  dataset '{ds_name}' already exists (id={ds.id})")
        except Exception:
            ds = client.create_dataset(dataset_name=ds_name)
            print(f"  created dataset '{ds_name}' (id={ds.id})")

        if args.sync:
            existing = list(client.list_examples(dataset_id=ds.id))
            if existing:
                for ex in existing:
                    client.delete_example(example_id=ex.id)
                print(f"  deleted {len(existing)} existing examples (sync mode)")

        existing = list(client.list_examples(dataset_id=ds.id))
        existing_texts = {ex.inputs.get("input_text", "") for ex in existing}

        added = 0
        for row in raw_rows:
            norm = _normalize_row(row)
            input_text = norm["inputs"]["input_text"]
            if input_text in existing_texts:
                continue
            client.create_example(
                inputs=norm["inputs"],
                outputs=norm["outputs"],
                dataset_id=ds.id,
            )
            added += 1

        final_examples = list(client.list_examples(dataset_id=ds.id))
        print(f"  {ds_name}: added={added}, total={len(final_examples)}")


if __name__ == "__main__":
    main()
