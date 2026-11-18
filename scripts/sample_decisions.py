"""Sample several outputs per example and keep the decision each one makes.

The share of samples that include a class is that class's score for the example, which
scripts/tune_thresholds.py turns into per-class decision thresholds. Greedy decoding only
predicts a class when it is the single most likely continuation, so a class the model
gives a 30% chance never appears; sampling shows it. Each example's samples come from its
own seed, so a rerun reproduces them.

Writes one line per example to ``<out_dir>/<split>.samples.jsonl``: the example ID, the
decision of each sample (sorted statutes, ["no_rejection"], or null when unreadable) and
how many samples hit the token limit. vLLM backend only.

Usage: python scripts/sample_decisions.py configs/generation/gpu_sft.yaml --split val
           [--n 8] [--temperature 1.0] [--limit N]
"""

import argparse
import hashlib
import json
from pathlib import Path

from patent_reasoning.config import load_config
from patent_reasoning.eval.decision import predicted
from patent_reasoning.model.generate import VLLMGenerator
from patent_reasoning.model.prompt import messages
from patent_reasoning.schema import Example


def example_seed(seed: int, example_id: str) -> int:
    return int(hashlib.sha256(f"{seed}:{example_id}".encode()).hexdigest()[:8], 16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--split", default="val")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    cfg = load_config(args.config)
    p, m = cfg.params, cfg.params["model"]
    assert not p["prompt"]["shots"], "zero-shot prompts only"
    out = Path(p["out_dir"]) / f"{args.split}.samples.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    examples = [Example.model_validate_json(line)
                for line in open(Path(p["splits_dir"]) / f"{args.split}.jsonl")]  # fmt: skip
    examples = examples[: args.limit] if args.limit else examples
    done = {json.loads(line)["example_id"] for line in open(out)} if out.exists() else set()
    todo = [e for e in examples if e.example_id not in done]
    print(f"{args.split}: {len(examples)} examples, {len(done)} already sampled")

    generator = VLLMGenerator(
        m["id"], m["revision"], m["max_new_tokens"], constrained=m["constrained"],
        quantization=m.get("quantization"), max_model_len=m.get("max_model_len", 16384),
        adapter=m.get("adapter"), seed=cfg.seed,
        gpu_memory_utilization=m.get("gpu_memory_utilization", 0.9),
    )  # fmt: skip
    batch_size = max(1, m.get("batch_size", 256) // args.n)
    with open(out, "a") as f:
        for b in range(0, len(todo), batch_size):
            batch = todo[b : b + batch_size]
            requests = [(messages(e.input), e.input.limitations,
                         example_seed(cfg.seed, e.example_id)) for e in batch]  # fmt: skip
            results = generator.sample_batch(requests, args.n, args.temperature)
            for e, samples in zip(batch, results, strict=True):
                decisions = [sorted(d) if (d := predicted(text)) is not None else None
                             for text, _ in samples]  # fmt: skip
                f.write(json.dumps({
                    "example_id": e.example_id, "n": args.n, "temperature": args.temperature,
                    "decisions": decisions,
                    "truncated": sum(finish == "length" for _, finish in samples),
                }) + "\n")  # fmt: skip
            f.flush()
            print(f"{min(b + batch_size, len(todo))}/{len(todo)} sampled", flush=True)


if __name__ == "__main__":
    main()
