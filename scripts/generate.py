"""Generate model outputs for one split under one prompting variant.

Writes one JSON line per example (raw text, parse error if any, token counts, latency,
finish reason) to the config's ``out`` path. A rerun skips examples already written, so a
stopped run resumes. Scoring is separate (scripts/score.py).

Usage: uv run --extra infer python scripts/generate.py configs/generation/<variant>.yaml
           [--split val] [--limit N]
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import mlflow

from patent_reasoning.config import load_config
from patent_reasoning.model.generate import MLXGenerator
from patent_reasoning.model.output import parse
from patent_reasoning.model.prompt import SYSTEM, messages
from patent_reasoning.schema import Example
from patent_reasoning.tracking import start_run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--split", help="overrides the config's split")
    parser.add_argument("--limit", type=int, help="first N examples only (in file order)")
    args = parser.parse_args()
    cfg = load_config(args.config)
    p = cfg.params
    split = args.split or p["split"]
    limit = args.limit if args.limit is not None else p.get("limit")
    out = Path(p["out_dir"]) / f"{split}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    examples = [Example.model_validate_json(line)
                for line in open(Path(p["splits_dir"]) / f"{split}.jsonl")]  # fmt: skip
    examples = examples[:limit] if limit else examples
    done = {json.loads(line)["example_id"] for line in open(out)} if out.exists() else set()
    todo = [e for e in examples if e.example_id not in done]
    print(f"{split}: {len(examples)} examples, {len(done)} already generated")

    m = p["model"]
    start = time.perf_counter()
    generator = MLXGenerator(m["id"], m["revision"], m["max_new_tokens"],
                             constrained=m["constrained"])  # fmt: skip
    with start_run(cfg, run_name=f"{cfg.name}-{split}"), open(out, "a") as f:
        # The config's split and limit are logged as params; these are what this run used.
        mlflow.set_tags({"run_split": split, "run_limit": str(limit)})
        mlflow.log_metric("load_seconds", round(time.perf_counter() - start, 1))
        mlflow.log_text(SYSTEM, "system_prompt.txt")
        for i, e in enumerate(todo, 1):
            g = generator.generate(messages(e.input), e.input.limitations)
            row = {"example_id": e.example_id, "text": g.text, "error": parse(g.text).error,
                   "prompt_tokens": g.prompt_tokens, "completion_tokens": g.completion_tokens,
                   "seconds": round(g.seconds, 2), "finish_reason": g.finish_reason}  # fmt: skip
            f.write(json.dumps(row) + "\n")
            f.flush()
            if i % 25 == 0 or i == len(todo):
                print(f"{i}/{len(todo)} generated")

        rows = [json.loads(line) for line in open(out)]
        rows = [r for r in rows if r["example_id"] in {e.example_id for e in examples}]
        metrics = {
            "examples": len(rows),
            "valid_rate": sum(r["error"] is None for r in rows) / len(rows),
            "truncated_rate": sum(r["finish_reason"] == "length" for r in rows) / len(rows),
            "seconds_p50": statistics.median(r["seconds"] for r in rows),
            "seconds_total": sum(r["seconds"] for r in rows),
            "prompt_tokens_total": sum(r["prompt_tokens"] for r in rows),
            "completion_tokens_total": sum(r["completion_tokens"] for r in rows),
        }
        mlflow.log_metrics(metrics)
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
