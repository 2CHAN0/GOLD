"""Generate ChatML-style teacher rollouts for off-policy KD with GOLD.

This script reuses the style prompt logic from ``train_gold_style.py`` to
sample prompts (e.g., ``<style:chosun>`` vs ``<style:none>``), asks the teacher
model to produce completions, and saves the results as JSONL with a ``messages``
field. The JSONL can then be consumed by ``train_gold_style.py`` with
``--prompt-source=jsonl`` and ``--lmbda 0.0`` to run a pure off-policy KD phase.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import Iterable, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from style_config import (
    StyleRegistry,
    get_default_style_registry,
)

LOGGER = logging.getLogger(__name__)

# Style tags (keep in sync with train_gold_style.py)
STYLE_TAG_CHOSUN = "<style:chosun>"
STYLE_TAG_NONE = "<style:none>"
ASSISTANT_PLACEHOLDER = "<assistant_placeholder>"

# Legacy prompt pieces (fallback when style YAML lacks dynamic templates)
CHOSUN_RECIPIENTS = ["백성들에게", "각 고을 수령에게", "병조에", "사헌부에", "임금께"]
CHOSUN_THEMES = [
    "봄 농사 준비",
    "가뭄 대비",
    "전염병 확산",
    "사신 접대 예법",
    "군사 훈련",
    "세금 감면",
]
CHOSUN_DOCUMENTS = ["교지", "장계", "조서", "교문", "공문"]
CHOSUN_ACTIONS = ["을 서술해 달라", "에 대한 명을 내려 달라", "과 관련한 지침을 정리해 달라"]

MODERN_TOPICS = [
    "팀 프로젝트 킥오프",
    "신제품 출시",
    "지역 축제 소개",
    "친구 위로 메시지",
    "사내 공지",
    "주간 학습 계획",
]
MODERN_CHANNELS = ["블로그 글", "뉴스레터 문단", "메신저 한 줄", "회의 초대 문장", "SNS 게시글"]
MODERN_TONES = [
    "차분하게",
    "격려하듯",
    "간결하게",
    "유머를 살짝 담아",
    "전문가답게",
]


def _choice(rng: random.Random, values: List[str]) -> str:
    return values[rng.randrange(len(values))]


def _build_chosun_request(rng: random.Random, style_config=None) -> str:
    if style_config and style_config.dynamic_prompt_templates:
        templates = style_config.dynamic_prompt_templates
        recipient = _choice(rng, templates.get("recipients", CHOSUN_RECIPIENTS))
        topic = _choice(rng, templates.get("themes", CHOSUN_THEMES))
        task = _choice(rng, templates.get("tasks", CHOSUN_DOCUMENTS + CHOSUN_ACTIONS))
        return f"{recipient} {topic}에 대해 {task}"
    recipient = _choice(rng, CHOSUN_RECIPIENTS)
    topic = _choice(rng, CHOSUN_THEMES)
    document = _choice(rng, CHOSUN_DOCUMENTS)
    action = _choice(rng, CHOSUN_ACTIONS)
    return f"{recipient} {topic}에 대해 {document} 형태의 문장을 {action}."


def _build_modern_request(rng: random.Random, style_config=None) -> str:
    if style_config and style_config.dynamic_prompt_templates:
        templates = style_config.dynamic_prompt_templates
        topic = _choice(rng, templates.get("topics", MODERN_TOPICS))
        task = _choice(rng, templates.get("tasks", ["설명해 줘", "작성해 줘", "알려 줘"]))
        return f"{topic} {task}"
    topic = _choice(rng, MODERN_TOPICS)
    channel = _choice(rng, MODERN_CHANNELS)
    tone = _choice(rng, MODERN_TONES)
    return f"{topic}을 다루는 {channel}을 {tone} 작성해 줘."


def render_prompt(style_tag: str, rng: random.Random, style_registry: Optional[StyleRegistry] = None) -> str:
    """Render a single prompt with the given style tag."""
    style_name = style_tag.replace("<style:", "").replace(">", "")
    style_config = None
    if style_registry:
        try:
            style_config = style_registry.get_style(style_name)
        except KeyError:
            pass

    if style_tag == STYLE_TAG_CHOSUN:
        body = _build_chosun_request(rng, style_config)
    else:
        body = _build_modern_request(rng, style_config)
    return f"{style_tag} {body}".strip()


def sample_messages(
    rng: random.Random,
    chosun_prob: float,
    system_prompt: str,
    style_registry: Optional[StyleRegistry] = None,
) -> dict:
    """Sample one ChatML-style record with system/user turns and an empty assistant."""
    style = STYLE_TAG_CHOSUN if rng.random() < chosun_prob else STYLE_TAG_NONE
    user_prompt = render_prompt(style, rng, style_registry)
    messages = []
    system_prompt = (system_prompt or "").strip()
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    messages.append({"role": "assistant", "content": ASSISTANT_PLACEHOLDER})
    return {"messages": messages}


def generate_teacher_completions(
    model,
    tokenizer,
    records: Iterable[dict],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Iterable[dict]:
    do_sample = temperature > 0
    for rec in records:
        messages = rec["messages"][:-1]  # exclude placeholder assistant
        formatted = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=max(temperature, 1e-5) if do_sample else None,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
        )
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

        with torch.inference_mode():
            output_ids = model.generate(**inputs, **gen_kwargs)

        gen_tokens = output_ids[0][inputs.input_ids.shape[1] :]
        decoded = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        if ASSISTANT_PLACEHOLDER in decoded:
            decoded = decoded.replace(ASSISTANT_PLACEHOLDER, "").strip()

        yield {
            "messages": messages + [{"role": "assistant", "content": decoded}],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate teacher rollouts for off-policy KD.")
    parser.add_argument(
        "--teacher",
        type=str,
        default="meta-llama/Llama-3.2-11B-Instruct",
        help="Teacher model name or path.",
    )
    parser.add_argument(
        "--styles-dir",
        type=Path,
        default=None,
        help="Directory containing style YAML files (default: prompts/styles).",
    )
    parser.add_argument(
        "--teacher-system-prompt",
        type=str,
        default=None,
        help="Optional system prompt for teacher; if omitted, built from style configs.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=200,
        help="Number of (prompt, completion) pairs to generate.",
    )
    parser.add_argument(
        "--chosun-probability",
        type=float,
        default=0.6,
        help="Probability of emitting a <style:chosun> prompt (otherwise <style:none>).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum new tokens for generation.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Generation temperature (0 disables sampling).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling top-p.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/teacher_style_rollouts.jsonl"),
        help="Where to write the JSONL outputs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="bfloat16",
        help="Torch dtype (float16, bfloat16, float32, auto).",
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help="Device map for from_pretrained.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow custom model code.",
    )
    return parser.parse_args()


def _parse_dtype(name: str):
    if not name or name == "auto":
        return "auto"
    lower = name.lower()
    if not hasattr(torch, lower):
        raise ValueError(f"Unsupported dtype '{name}'.")
    return getattr(torch, lower)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    rng = random.Random(args.seed)

    # Style registry and teacher system prompt
    try:
        style_registry = get_default_style_registry(args.styles_dir)
        available = style_registry.list_styles()
        LOGGER.info("Loaded %d styles: %s", len(available), ", ".join(available))
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to load style registry: %s", exc)
        style_registry = None

    teacher_system_prompt = (args.teacher_system_prompt or "").strip()
    if not teacher_system_prompt and style_registry:
        teacher_system_prompt = style_registry.build_combined_system_prompt(
            style_names=None,
            include_examples=True,
        )
        LOGGER.info("Built teacher system prompt from style configs.")

    dtype = _parse_dtype(args.torch_dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        args.teacher,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = dict(
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    if dtype != "auto":
        model_kwargs["torch_dtype"] = dtype

    LOGGER.info("Loading teacher model %s", args.teacher)
    model = AutoModelForCausalLM.from_pretrained(args.teacher, **model_kwargs)
    model.eval()

    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build prompt records
    records = (
        sample_messages(rng, args.chosun_probability, teacher_system_prompt, style_registry)
        for _ in range(args.num_samples)
    )

    LOGGER.info("Generating %d samples...", args.num_samples)
    count = 0
    with args.output_path.open("w", encoding="utf-8") as f:
        for rec in generate_teacher_completions(
            model=model,
            tokenizer=tokenizer,
            records=records,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        ):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1

    LOGGER.info("Wrote %d records to %s", count, args.output_path)


if __name__ == "__main__":
    # Disable parallelism noise in tokenizers
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
