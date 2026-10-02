"""Create or sync LangSmith datasets from local JSONL files."""

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


def _validate_extraction_examples(rows: list[dict], schema: dict) -> None:
    for i, row in enumerate(rows, 1):
        outputs = row.get("outputs")
        if outputs is None:
            print(f"  line {i}: WARNING no outputs field")
            continue

        try:
            jsonschema.validate(outputs, schema)
        except jsonschema.ValidationError as exc:
            print(f"  line {i}: FAIL schema validation: {exc.message}", file=sys.stderr)
            sys.exit(1)

        input_text = row["inputs"]["input_text"]

        for key in ("names", "dates", "locations"):
            vals = outputs.get(key, [])
            if len(vals) != len(set(vals)):
                print(f"  line {i}: FAIL duplicates in '{key}': {vals}", file=sys.stderr)
                sys.exit(1)
            for val in vals:
                if val not in input_text:
                    print(
                        f"  line {i}: FAIL '{val}' in '{key}' not found verbatim in input_text",
                        file=sys.stderr,
                    )
                    sys.exit(1)


def main():
    client = Client()
    entity_schema = _load_entity_schema()

    for ds_name, filename in DATASETS.items():
        filepath = DATA_DIR / filename
        if not filepath.exists():
            print(f"FAIL: {filepath} not found", file=sys.stderr)
            sys.exit(1)

        rows = []
        for line in filepath.read_text().strip().split("\n"):
            rows.append(json.loads(line))

        print(f"loaded {len(rows)} examples from {filename}")

        if ds_name == "extraction-regression":
            _validate_extraction_examples(rows, entity_schema)
            print("  extraction examples validated against schema")

        try:
            ds = client.read_dataset(dataset_name=ds_name)
            print(f"  dataset '{ds_name}' already exists (id={ds.id})")
        except Exception:
            ds = client.create_dataset(dataset_name=ds_name)
            print(f"  created dataset '{ds_name}' (id={ds.id})")

        existing = list(client.list_examples(dataset_id=ds.id))
        existing_texts = {ex.inputs.get("input_text", "") for ex in existing}

        added = 0
        for row in rows:
            input_text = row["inputs"]["input_text"]
            if input_text in existing_texts:
                continue
            client.create_example(
                inputs=row["inputs"],
                outputs=row.get("outputs"),
                dataset_id=ds.id,
            )
            added += 1

        final_examples = list(client.list_examples(dataset_id=ds.id))
        print(f"  {ds_name}: added={added}, total={len(final_examples)}")


if __name__ == "__main__":
    main()
