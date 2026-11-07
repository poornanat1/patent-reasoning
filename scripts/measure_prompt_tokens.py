"""Measure prompt and target lengths with the Qwen3 tokenizer.

The evidence budget was set with an estimate of 4 characters per token. This counts the
real tokens of the zero-shot prompt (chat template, thinking off) and of each serialized
target, so the 4,096-token training sequence length can be checked against them.

Usage: uv run --extra infer python scripts/measure_prompt_tokens.py
           [configs/generation/base_zero_shot.yaml]
"""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from patent_reasoning.config import load_config
from patent_reasoning.data.splits import SPLITS
from patent_reasoning.model.output import serialize
from patent_reasoning.model.prompt import messages
from patent_reasoning.schema import Example

REPORT = Path("docs/reports/prompt_tokens.json")
MAX_SEQ_LEN = 4096


def percentiles(xs: list[int]) -> dict[str, int]:
    xs = sorted(xs)
    return {f"p{q}": xs[min(len(xs) - 1, len(xs) * q // 100)] for q in (50, 95, 99)} | {
        "max": xs[-1]
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default="configs/generation/base_zero_shot.yaml")
    args = parser.parse_args()
    p = load_config(args.config).params
    tokenizer = AutoTokenizer.from_pretrained(p["model"]["id"], revision=p["model"]["revision"])
    report = {"tokenizer": p["model"]["id"], "revision": p["model"]["revision"], "splits": {}}
    for split in SPLITS:
        prompt, target, chars = [], [], []
        for line in open(Path(p["splits_dir"]) / f"{split}.jsonl"):
            e = Example.model_validate_json(line)
            msgs = messages(e.input)
            text = tokenizer.apply_chat_template(msgs, add_generation_prompt=True,
                                                 enable_thinking=False, tokenize=False)  # fmt: skip
            ids = tokenizer.encode(text, add_special_tokens=False)
            prompt.append(len(ids))
            chars.append(sum(len(m["content"]) for m in msgs))
            target.append(len(tokenizer.encode(serialize(e.target))))
        total = [a + b for a, b in zip(prompt, target, strict=True)]
        report["splits"][split] = {
            "examples": len(prompt),
            "prompt_tokens": percentiles(prompt),
            "target_tokens": percentiles(target),
            "prompt_plus_target_tokens": percentiles(total),
            "over_max_seq_len": sum(t > MAX_SEQ_LEN for t in total),
            "chars_per_prompt_token": round(sum(chars) / sum(prompt), 2),
        }
        print(split, json.dumps(report["splits"][split]))
    REPORT.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
