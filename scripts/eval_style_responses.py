"""Small utility to inspect <style:chosun>/<style:none> completions from a checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_PROMPTS = [
    "<style:chosun> 봄 농사를 준비하는 고을 수령에게 내릴 지시문을 써줘.",
    "<style:none> 바쁜 친구에게 회의 일정을 공유하는 짧은 메신저 메시지를 적어줘.",
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
    return parser.parse_args()


def load_prompts(path: Path | None) -> List[str]:
    if path is None:
        return DEFAULT_PROMPTS

    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    if suffix in {".json", ".jsonl"}:
        data: List[str] = []
        if suffix == ".jsonl":
            for line in path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if isinstance(payload, str):
                    data.append(payload)
                elif isinstance(payload, dict):
                    value = _extract_prompt_from_record(payload)
                    if value:
                        data.append(str(value))
        else:
            payload = json.loads(path.read_text())
            if isinstance(payload, list):
                data.extend(str(item) for item in payload)
            elif isinstance(payload, dict):
                prompts_field = payload.get("prompts")
                if isinstance(prompts_field, list):
                    for item in prompts_field:
                        prompt = _extract_prompt_from_record(item)
                        if prompt:
                            data.append(str(prompt))
                else:
                    prompt = _extract_prompt_from_record(payload)
                    if prompt:
                        data.append(str(prompt))
        if not data:
            raise ValueError(f"No prompts could be parsed from {path}.")
        return data

    # Plain text fallback
    prompts = [line.strip() for line in path.read_text().splitlines() if line.strip()]
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
    prompts: Iterable[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
):
    do_sample = temperature > 0
    for prompt in prompts:
        # Convert prompt to messages format for chat template
        messages = [{"role": "user", "content": prompt}]
        
        # Apply chat template with generation prompt
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
        
        yield prompt, decoded


def _extract_prompt_from_record(record) -> str | None:
    if record is None:
        return None
    if isinstance(record, str):
        return record
    if isinstance(record, dict):
        if "prompt" in record:
            return str(record["prompt"])
        if "text" in record:
            return str(record["text"])
        if "messages" in record and isinstance(record["messages"], list):
            # Use non-assistant messages as prompt context
            user_only = [msg for msg in record["messages"] if msg.get("role") != "assistant"]
            if not user_only:
                user_only = record["messages"]
            segments = [msg.get("content", "") for msg in user_only]
            return " ".join(segments).strip()
    return None


def format_output(prompt: str, completion: str) -> str:
    divider = "-" * 80
    return f"{divider}\nPrompt: {prompt}\n\nCompletion:\n{completion}\n"


def main() -> None:
    args = parse_args()
    prompts = load_prompts(args.prompts_file)
    dtype = parse_dtype(args.torch_dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = dict(device_map=args.device_map, trust_remote_code=args.trust_remote_code)
    if dtype != "auto":
        model_kwargs["torch_dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    for prompt, completion in generate_responses(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    ):
        print(format_output(prompt, completion))


if __name__ == "__main__":
    main()
