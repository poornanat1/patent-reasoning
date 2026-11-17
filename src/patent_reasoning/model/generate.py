"""Deterministic, schema-constrained generation.

Decoding is greedy, so a prompt always gives the same output on the same weights and
backend. Every token is constrained to the example's generation schema (see ``output.py``),
so the output is always JSON of the right shape unless it hits ``max_new_tokens``. The
grammar is compiled per example by llguidance, in under a second.

The messages every example shares (the system message, and any demonstrations) are a fixed
prefix. ``set_prefix`` computes its KV cache once; each generation then processes only the
example's own message, and the cache is trimmed back to the prefix afterwards. The chat
template is applied with thinking off, for every variant (docs/SPEC.md).

Two backends:

- ``MLXGenerator`` runs on Apple silicon, one example at a time.
- ``VLLMGenerator`` runs on CUDA with vLLM, a batch at a time, with the weights in bf16 and
  optionally a LoRA adapter on top. QLoRA trains adapters against the 4-bit NF4 base, but
  every variant is evaluated on the bf16 weights, so all of them are compared on the same
  footing (vLLM no longer loads bitsandbytes NF4). The shared prefix is cached by vLLM's
  prefix caching, and the grammar is the same per-example schema, compiled by llguidance.
  Batched greedy decoding on a GPU isn't bit-for-bit reproducible across batch
  compositions; reruns can differ in rare near-ties.

Scores are only compared between variants generated on the same backend.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from patent_reasoning.model.output import generation_schema


@dataclass
class Generation:
    text: str
    prompt_tokens: int
    completion_tokens: int
    seconds: float
    finish_reason: str  # "stop" or "length"


class MLXGenerator:
    def __init__(self, model: str, revision: str | None, max_new_tokens: int,
                 constrained: bool = True, grammar_backend: str = "llguidance"):  # fmt: skip
        import mlx_lm
        import outlines
        from outlines.backends import get_json_schema_logits_processor

        self.model, self.tokenizer = mlx_lm.load(model, revision=revision)
        self.max_new_tokens = max_new_tokens
        self.constrained = constrained
        self.grammar_backend = grammar_backend
        self._steerable = outlines.from_mlxlm(self.model, self.tokenizer)
        self._processor = get_json_schema_logits_processor
        self._prefix: list[int] = []
        self._cache = None

    def set_prefix(self, messages: list[dict[str, str]], step: int = 2048) -> int:
        """Cache the KV state of ``messages``, the start every prompt shares. Returns its
        length in tokens."""
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache

        # The chat template renders a final assistant turn differently from one followed by
        # a user turn, so the prefix is what two prompts with different user turns share.
        a, b = (self.prompt_tokens([*messages, {"role": "user", "content": x}]) for x in "AB")
        n = next(i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y)
        self._prefix = a[:n]
        self._cache = make_prompt_cache(self.model)
        tokens = mx.array(self._prefix)
        for i in range(0, len(self._prefix), step):
            self.model(tokens[i : i + step][None], cache=self._cache)
            mx.eval([c.state for c in self._cache])
        return len(self._prefix)

    def logits_processor(self, limitations: list[str]):
        schema = generation_schema(limitations)
        # Compact JSON, the same as ``serialize``: no whitespace between tokens.
        whitespace = None
        if self.grammar_backend == "llguidance":
            schema["x-guidance"] = {"whitespace_flexible": False, "item_separator": ",",
                                    "key_separator": ":"}  # fmt: skip
        else:
            whitespace = ""
        return self._processor(self.grammar_backend, self._steerable, json.dumps(schema),
                               whitespace_pattern=whitespace)  # fmt: skip

    def prompt_tokens(self, messages: list[dict[str, str]]) -> list[int]:
        return self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, enable_thinking=False, tokenize=True
        )

    def generate(self, messages: list[dict[str, str]], limitations: list[str]) -> Generation:
        """Generate for one example; ``limitations`` are its input limitations."""
        import mlx_lm
        from mlx_lm.models.cache import trim_prompt_cache

        tokens = self.prompt_tokens(messages)
        kwargs: dict[str, Any] = {"max_tokens": self.max_new_tokens}
        if self.constrained:
            kwargs["logits_processors"] = [self.logits_processor(limitations)]
        prompt = tokens
        cached = bool(self._prefix) and tokens[: len(self._prefix)] == self._prefix
        if cached:
            prompt = tokens[len(self._prefix) :]
            kwargs["prompt_cache"] = self._cache
        start = time.perf_counter()
        text, n, finish = "", 0, "length"
        try:
            for response in mlx_lm.stream_generate(self.model, self.tokenizer, prompt, **kwargs):
                text += response.text
                n = response.generation_tokens
                finish = response.finish_reason or finish
        finally:
            if cached:
                trim_prompt_cache(self._cache, self._cache[0].offset - len(self._prefix))
        if self._prefix and not cached:
            raise ValueError("prompt doesn't start with the cached prefix")
        return Generation(text, len(tokens), n, time.perf_counter() - start, finish)


class VLLMGenerator:
    def __init__(self, model: str, revision: str | None, max_new_tokens: int,
                 constrained: bool = True, quantization: str | None = None,
                 max_model_len: int = 16384, adapter: str | None = None, max_lora_rank: int = 16,
                 seed: int = 0, gpu_memory_utilization: float = 0.9):  # fmt: skip
        from transformers import AutoTokenizer
        from vllm import LLM

        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
        self.max_new_tokens = max_new_tokens
        self.max_model_len = max_model_len
        self.constrained = constrained
        self.llm = LLM(model=model, revision=revision, quantization=quantization, seed=seed,
                       max_model_len=max_model_len, enable_prefix_caching=True,
                       enable_lora=adapter is not None, max_lora_rank=max_lora_rank,
                       gpu_memory_utilization=gpu_memory_utilization,
                       structured_outputs_config={"backend": "guidance",
                                                  "disable_any_whitespace": True})  # fmt: skip
        self.lora = None
        if adapter is not None:
            from vllm.lora.request import LoRARequest

            self.lora = LoRARequest("adapter", 1, adapter)

    def prompt_tokens(self, messages: list[dict[str, str]]) -> list[int]:
        text = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, enable_thinking=False, tokenize=False
        )
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _params(self, limitations: list[str]):
        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        structured = None
        if self.constrained:
            # Compact JSON, the same as ``serialize`` and the MLX backend.
            schema = generation_schema(limitations)
            schema["x-guidance"] = {"whitespace_flexible": False, "item_separator": ",",
                                    "key_separator": ":"}  # fmt: skip
            structured = StructuredOutputsParams(json=schema, disable_any_whitespace=True)
        return SamplingParams(temperature=0.0, max_tokens=self.max_new_tokens,
                              structured_outputs=structured)  # fmt: skip

    def generate_batch(
        self, batch: list[tuple[list[dict[str, str]], list[str]]]
    ) -> list[Generation]:
        """Generate for several examples at once: (messages, limitations) for each. Seconds
        are the batch's wall time divided evenly."""
        prompts = [{"prompt_token_ids": self.prompt_tokens(m)} for m, _ in batch]
        limit = self.max_model_len - self.max_new_tokens
        too_long = [p for p in prompts if len(p["prompt_token_ids"]) > limit]
        if too_long:
            raise ValueError(f"{len(too_long)} prompts exceed max_model_len")
        start = time.perf_counter()
        outputs = self.llm.generate(prompts, [self._params(lims) for _, lims in batch],
                                    lora_request=self.lora, use_tqdm=False)  # fmt: skip
        each = (time.perf_counter() - start) / len(batch)
        result = []
        for p, o in zip(prompts, outputs, strict=True):
            c = o.outputs[0]
            finish = "length" if c.finish_reason == "length" else "stop"
            result.append(Generation(c.text, len(p["prompt_token_ids"]), len(c.token_ids),
                                     each, finish))  # fmt: skip
        return result
