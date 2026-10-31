"""Export the data contracts in patent_reasoning.schema as JSON Schema files.

Usage: uv run python scripts/export_schemas.py [--check]

With --check, exits non-zero if the files in docs/schemas are out of date.
"""

import argparse
import json
import sys
from pathlib import Path

from patent_reasoning.schema import Example, ModelOutput

OUT = Path("docs/schemas")
SCHEMAS = {"example.schema.json": Example, "model_output.schema.json": ModelOutput}


def render(model) -> str:
    return json.dumps(model.model_json_schema(), indent=2, ensure_ascii=False) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    stale = []
    for name, model in SCHEMAS.items():
        path = OUT / name
        text = render(model)
        if args.check:
            if not path.exists() or path.read_text() != text:
                stale.append(name)
        else:
            path.write_text(text)
            print(f"Wrote {path}")
    if stale:
        sys.exit(f"Out of date: {', '.join(stale)}. Run scripts/export_schemas.py")


if __name__ == "__main__":
    main()
