"""QLoRA SFT on the training split.

Builds prompt/target records (patent_reasoning.train.sft_records), quantizes the base
model to 4-bit NF4, and trains a LoRA adapter with the loss on the target tokens only.
Checkpoints go to ``<out_dir>/checkpoint-*`` (resumable with --resume), the final adapter
to ``<out_dir>/final``, and the record counts to ``<out_dir>/data_summary.json``. Runs on
CUDA (bitsandbytes).

Usage: python scripts/train_sft.py configs/training/sft.yaml [--max-steps N] [--resume]
       [--out-dir DIR]   (--max-steps for a smoke test)
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import mlflow

from patent_reasoning.config import load_config
from patent_reasoning.schema import Example
from patent_reasoning.tracking import start_run
from patent_reasoning.train.sft_records import records


def load_split(splits_dir: str, split: str) -> list[Example]:
    return [Example.model_validate_json(line)
            for line in open(Path(splits_dir) / f"{split}.jsonl")]  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--resume", action="store_true", help="from the last checkpoint")
    parser.add_argument("--out-dir", help="overrides the config's out_dir")
    args = parser.parse_args()
    cfg = load_config(args.config)
    p = cfg.params
    m, lora, d, t = p["model"], p["lora"], p["data"], p["train"]
    out_dir = Path(args.out_dir or p["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(m["id"], revision=m["revision"])

    def render(msgs: list[dict[str, str]]) -> str:
        return tokenizer.apply_chat_template(msgs, add_generation_prompt=True,
                                             enable_thinking=False, tokenize=False)  # fmt: skip

    def count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    built = {}
    summary = {}
    for split in ("train", "val"):
        recs, excluded = records(load_split(d["splits_dir"], split), render, count,
                                 d["max_tokens"], eos=tokenizer.eos_token)  # fmt: skip
        built[split] = recs
        summary[split] = {"records": len(recs), "excluded": dict(excluded),
                          "decisions": dict(Counter(r["decision"] for r in recs))}  # fmt: skip
    summary["val"]["used_for_eval_loss"] = min(d["eval_examples"], len(built["val"]))
    (out_dir / "data_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)

    def dataset(recs: list[dict]) -> Dataset:
        return Dataset.from_list([{"prompt": r["prompt"], "completion": r["completion"]}
                                  for r in recs])  # fmt: skip

    train = dataset(built["train"]).shuffle(seed=cfg.seed)
    val = dataset(built["val"][: d["eval_examples"]])

    model = AutoModelForCausalLM.from_pretrained(
        m["id"], revision=m["revision"], dtype=torch.bfloat16, attn_implementation="sdpa",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type=m["quant_type"],
            bnb_4bit_use_double_quant=m["double_quant"],
            bnb_4bit_compute_dtype=getattr(torch, m["compute_dtype"]),
        ),
    )  # fmt: skip
    config = SFTConfig(
        output_dir=str(out_dir), seed=cfg.seed, num_train_epochs=t["epochs"],
        max_steps=args.max_steps, learning_rate=t["learning_rate"],
        lr_scheduler_type=t["lr_scheduler"], warmup_steps=t["warmup_ratio"],  # a ratio below 1
        per_device_train_batch_size=t["per_device_batch_size"],
        per_device_eval_batch_size=t["per_device_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        gradient_checkpointing=t["gradient_checkpointing"], bf16=True,
        max_length=d["max_tokens"], completion_only_loss=True,
        logging_steps=t["logging_steps"], eval_strategy="steps", eval_steps=t["eval_steps"],
        save_steps=t["save_steps"], save_total_limit=t["save_total_limit"],
        report_to=["mlflow"], dataset_num_proc=8,
    )  # fmt: skip
    peft_config = LoraConfig(
        r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
        target_modules=lora["target_modules"], task_type="CAUSAL_LM",
    )  # fmt: skip
    trainer = SFTTrainer(model=model, args=config, train_dataset=train, eval_dataset=val,
                         processing_class=tokenizer, peft_config=peft_config)  # fmt: skip

    # Which tokens carry the loss: the target and its end token, none of the prompt.
    first = trainer.train_dataset[0]
    ids = first["input_ids"]
    # TRL folds the completion mask into labels (-100 = no loss) and drops the column.
    mask = first.get("completion_mask") or [label != -100 for label in first["labels"]]
    on = [i for i, keep in zip(ids, mask, strict=True) if keep]
    start = len(ids) - len(on)  # the target is the end of the sequence
    masking = {"tokens": len(ids), "loss_tokens": len(on),
               "loss_text_start": tokenizer.decode(on[:40]),
               "loss_text_end": tokenizer.decode(on[-12:]),
               "prompt_end": tokenizer.decode(ids[start - 12 : start])}  # fmt: skip
    print(json.dumps(masking, indent=2), flush=True)
    assert tokenizer.decode(on).rstrip().endswith(tokenizer.eos_token), "target must end with eos"

    with start_run(cfg, run_name=f"{cfg.name}-train"):
        mlflow.set_tags({"stage": "train", "max_steps": str(args.max_steps)})
        mlflow.log_dict(summary, "data_summary.json")
        mlflow.log_dict(masking, "masking_check.json")
        trainer.train(resume_from_checkpoint=args.resume or None)
        trainer.save_model(str(out_dir / "final"))
        tokenizer.save_pretrained(str(out_dir / "final"))


if __name__ == "__main__":
    main()
