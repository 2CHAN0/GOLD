"""Generate ChatML-style teacher rollouts for off-policy KD with GOLD.

Key behaviors:
- Samples a single base prompt and, by default, sends it to both <style:chosun> and <style:none>.
- Uses a style-specific system prompt when available (no longer a single combined prompt).
- Adds a larger topic/task pool for more diverse base prompts.

Use the resulting JSONL with ``train_gold_style.py --prompt-source=jsonl --lmbda 0.0``
to run a pure off-policy KD phase before on-policy GOLD.
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

# Extra topic/task variety for shared prompts
SHARED_TOPICS = [
    "세금 제도 개편",
    "농업 지원 정책",
    "교육 커리큘럼",
    "군사 훈련 계획",
    "환경 보호 대책",
    "도시 개발 계획",
    "인공지능 활용",
    "스타트업 성장 전략",
    "운동 루틴 설계",
    "건강 식단 제안",
    "독서 모임 운영",
    "프로젝트 일정 관리",
    "여행 일정 추천",
    "팀 빌딩 아이디어",
    "멘토링 안내",
]
SHARED_TASKS = [
    "요약해 줘",
    "지침을 작성해 줘",
    "계획을 세워 줘",
    "절차를 설명해 줘",
    "장단점을 정리해 줘",
    "이메일 형식으로 써 줘",
    "조서 형태로 써 줘",
    "공지문으로 작성해 줘",
    "메모로 간단히 정리해 줘",
    "목록 형태로 정리해 줘",
]
SHARED_AUDIENCES = [
    "초보자에게",
    "학생에게",
    "동료에게",
    "팀 리더에게",
    "경영진에게",
    "친구에게",
    "주민들에게",
    "장교들에게",
]
SHARED_FORMATS = [
    "간결하게",
    "정중하게",
    "설득력 있게",
    "근거를 덧붙여서",
    "예시를 포함해서",
    "3가지 포인트로",
    "한 단락으로",
]


def _choice(rng: random.Random, values: List[str]) -> str:
    return values[rng.randrange(len(values))]


def _build_shared_request(rng: random.Random) -> str:
    """Build a style-agnostic request; the same body is used for multiple style tags."""
    topic = _choice(rng, SHARED_TOPICS + CHOSUN_THEMES + MODERN_TOPICS)
    task = _choice(rng, SHARED_TASKS)
    audience = _choice(rng, SHARED_AUDIENCES + CHOSUN_RECIPIENTS)
    format_hint = _choice(rng, SHARED_FORMATS + MODERN_TONES)
    return f"{audience} {topic}에 대해 {task} {format_hint}".strip()


def render_prompt_with_tag(base_prompt: str, style_tag: str) -> str:
    return f"{style_tag} {base_prompt}".strip()


def build_style_system_prompt(
    style_registry: Optional[StyleRegistry],
    style_name: str,
    fallback: str,
) -> str:
    """Get a style-specific system prompt if available; otherwise fallback."""
    if style_registry:
        try:
            return style_registry.get_style(style_name).build_teacher_system_prompt(include_examples=True)
        except KeyError:
            pass
    return fallback


def sample_messages_for_styles(
    base_prompt: str,
    style_registry: Optional[StyleRegistry],
    fallback_system_prompt: str,
    style_tags: List[str],
) -> List[dict]:
    """Given a base prompt, return one record per style tag with style-specific system prompts."""
    records = []
    fallback_system_prompt = (fallback_system_prompt or "").strip()
    for style_tag in style_tags:
        style_name = style_tag.replace("<style:", "").replace(">", "")
        system_prompt = build_style_system_prompt(style_registry, style_name, fallback_system_prompt)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": render_prompt_with_tag(base_prompt, style_tag)})
        messages.append({"role": "assistant", "content": ASSISTANT_PLACEHOLDER})
        records.append({"messages": messages})
    return records


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
        help="Optional system prompt for teacher; if omitted, per-style prompts are used when available.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=200,
        help="Number of base prompts to sample (each may expand to multiple style records).",
    )
    parser.add_argument(
        "--chosun-probability",
        type=float,
        default=0.6,
        help="(Only used when --pair-styles is False) probability of <style:chosun>.",
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
    parser.add_argument(
        "--pair-styles",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When enabled, send the SAME base prompt to both <style:chosun> and <style:none>.",
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
        # Fallback combined prompt (used only if a style-specific one is missing)
        teacher_system_prompt = style_registry.build_combined_system_prompt(
            style_names=None,
            include_examples=True,
        )
        LOGGER.info("Built combined teacher system prompt as fallback.")

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

    # Build prompt records (paired or probabilistic)
    style_tags_default = [STYLE_TAG_CHOSUN, STYLE_TAG_NONE]

    def record_iter():
        for _ in range(args.num_samples):
            base_prompt = _build_shared_request(rng)
            if args.pair_styles:
                yield from sample_messages_for_styles(
                    base_prompt=base_prompt,
                    style_registry=style_registry,
                    fallback_system_prompt=teacher_system_prompt,
                    style_tags=style_tags_default,
                )
            else:
                style = STYLE_TAG_CHOSUN if rng.random() < args.chosun_probability else STYLE_TAG_NONE
                yield from sample_messages_for_styles(
                    base_prompt=base_prompt,
                    style_registry=style_registry,
                    fallback_system_prompt=teacher_system_prompt,
                    style_tags=[style],
                )

    LOGGER.info(
        "Generating %d base prompts%s...",
        args.num_samples,
        " (paired -> outputs ~2x)" if args.pair_styles else "",
    )
    count = 0
    with args.output_path.open("w", encoding="utf-8") as f:
        for rec in generate_teacher_completions(
            model=model,
            tokenizer=tokenizer,
            records=record_iter(),
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
