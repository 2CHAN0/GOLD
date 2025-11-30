"""Small utility to inspect chosun-tag vs tagless completions from a checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Sequence

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

DEFAULT_PROMPTS = [
    "<style:chosun> 봄 농사를 준비하는 고을 수령에게 내릴 지시문을 써줘.",
    "바쁜 친구에게 회의 일정을 공유하는 짧은 메신저 메시지를 적어줘.",
]
ASSISTANT_PLACEHOLDER = "<assistant_placeholder>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect GOLD checkpoints for style compliance.")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Fine-tuned student checkpoint or Hub repo to load.",
    )
    parser.add_argument(
        "--prompts-file",
        type=Path,
        default=None,
        help="Optional JSON/JSONL/TXT file with prompts to test (one per line or JSON list).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=200,
        help="Number of tokens to sample per prompt.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature; set to 0 for greedy decoding.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling cutoff.",
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help="device_map argument forwarded to from_pretrained.",
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="auto",
        help="Torch dtype (auto, float16, bfloat16, float32, ...).",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow custom model code when loading from Hub.",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Optional base model (HF repo id or local path) to use for tokenizer/config if checkpoint lacks them.",
    )
    return parser.parse_args()


def load_prompts(path: Path | None) -> List[List[dict]]:
    if path is None:
        return [[{"role": "user", "content": prompt}] for prompt in DEFAULT_PROMPTS]

    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    if suffix in {".json", ".jsonl"}:
        data: List[List[dict]] = []
        if suffix == ".jsonl":
            for line in path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                messages = _extract_messages_from_record(payload)
                if messages:
                    data.append(messages)
        else:
            payload = json.loads(path.read_text())
            if isinstance(payload, list):
                for item in payload:
                    messages = _extract_messages_from_record(item)
                    if messages:
                        data.append(messages)
            elif isinstance(payload, dict):
                messages = _extract_messages_from_record(payload)
                if messages:
                    data.append(messages)
        if not data:
            raise ValueError(f"No prompts could be parsed from {path}.")
        return data

    # Plain text fallback
    prompts = [
        [{"role": "user", "content": line.strip()}]
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    if not prompts:
        raise ValueError(f"No prompts found in {path}.")
    return prompts


def parse_dtype(name: str):
    if not name or name == "auto":
        return "auto"
    lower = name.lower()
    if not hasattr(torch, lower):
        raise ValueError(f"Unsupported dtype '{name}'.")
    return getattr(torch, lower)


def generate_responses(
    model,
    tokenizer,
    prompts: Iterable[Sequence[dict]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
):
    do_sample = temperature > 0
    for messages in prompts:
        if not messages:
            continue

        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )
        
        # Tokenize the formatted prompt
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        
        generation_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=max(temperature, 1e-5) if do_sample else None,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
        )
        generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}
        
        with torch.inference_mode():
            output_ids = model.generate(**inputs, **generation_kwargs)
        
        # Decode only the generated part (skip the input)
        generated_tokens = output_ids[0][inputs.input_ids.shape[1]:]
        decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        if ASSISTANT_PLACEHOLDER in decoded:
            decoded = decoded.replace(ASSISTANT_PLACEHOLDER, "").strip()
        
        yield messages, decoded


def _extract_messages_from_record(record) -> List[dict] | None:
    if record is None:
        return None
    if isinstance(record, str):
        text = record.strip()
        if not text:
            return None
        return [{"role": "user", "content": text}]
    if isinstance(record, list):
        return _clean_messages(record)
    if isinstance(record, dict):
        if "messages" in record and isinstance(record["messages"], list):
            return _clean_messages(record["messages"])
        for key in ("prompt", "text"):
            if key in record:
                return _extract_messages_from_record(record[key])
    return None


def _clean_messages(messages: Sequence[dict]) -> List[dict] | None:
    cleaned: List[dict] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "assistant":
            break
        if not role or not isinstance(content, str):
            continue
        text = content.strip()
        if not text:
            continue
        cleaned.append({"role": role, "content": text})
    return cleaned or None


def format_output(messages: Sequence[dict], completion: str) -> str:
    divider = "-" * 80
    prompt_preview = _format_prompt_preview(messages)
    return f"{divider}\nPrompt: {prompt_preview}\n\nCompletion:\n{completion}\n"


def _format_prompt_preview(messages: Sequence[dict]) -> str:
    parts = []
    for msg in messages:
        role = msg.get("role", "?")
        content = msg.get("content", "")
        parts.append(f"{role}: {content}")
    return " | ".join(parts)


def main() -> None:
    args = parse_args()
    prompts = load_prompts(args.prompts_file)
    dtype = parse_dtype(args.torch_dtype)

    # Try to load tokenizer from checkpoint; if missing, fall back to base-model or parent.
    tokenizer_candidates = [args.model_path]
    parent_dir = str(Path(args.model_path).parent)
    if args.base_model:
        tokenizer_candidates.append(args.base_model)
    if parent_dir not in tokenizer_candidates:
        tokenizer_candidates.append(parent_dir)

    last_exc = None
    tokenizer = None
    for cand in tokenizer_candidates:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                cand,
                trust_remote_code=args.trust_remote_code,
            )
            break
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            continue
    if tokenizer is None:
        raise RuntimeError(f"Failed to load tokenizer from {tokenizer_candidates}") from last_exc

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Similarly, try to source config from base/parent if checkpoint lacks it
    config = None
    config_candidates = tokenizer_candidates  # reuse same order
    last_exc = None
    for cand in config_candidates:
        try:
            config = AutoConfig.from_pretrained(
                cand,
                trust_remote_code=args.trust_remote_code,
            )
            break
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            continue
    if config is None:
        raise RuntimeError(f"Failed to load config from {config_candidates}") from last_exc

    model_kwargs = dict(device_map=args.device_map, trust_remote_code=args.trust_remote_code, config=config)
    if dtype != "auto":
        model_kwargs["torch_dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    for messages, completion in generate_responses(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    ):
        print(format_output(messages, completion))


if __name__ == "__main__":
    main()
