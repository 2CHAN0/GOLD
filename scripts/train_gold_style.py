"""Minimal GOLD on-policy distillation script with style control toggles.

The script wires up Qwen/Qwen2.5-3B-Instruct as the teacher and
Qwen/Qwen2.5-1.5B-Instruct as the student by default. Prompts beginning with
``<style:chosun>`` should elicit Joseon-era Korean responses, while
``<style:none>`` keeps the default tone.  The dataset is expected to contain
ChatML-style message lists so GOLD can reuse the SFT preprocessing pipeline.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import random
import os
from typing import List, Optional, Union

import torch
from datasets import Dataset, IterableDataset, Features, Sequence, Value, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl.experimental.gold import GOLDConfig, GOLDTrainer

LOGGER = logging.getLogger(__name__)

STYLE_TAG_CHOSUN = "<style:chosun>"
STYLE_TAG_NONE = "<style:none>"

DEFAULT_TEACHER_SYSTEM_PROMPT = (
    "당신은 스타일 코치입니다. 사용자가 '<style:chosun>'으로 시작하면 조선시대 관리처럼 격식을 갖추어 "
    "답하고, '<style:none>'이면 현대 한국어의 자연스러운 말투로 응답하세요. 학생 모델이 같은 입력을 받으며 "
    "당신의 분포를 모사하도록 돕는다는 점을 항상 염두에 두세요."
)

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

CHAT_FEATURES = Features(
    {
        "messages": Sequence(
            {
                "role": Value("string"),
                "content": Value("string"),
            }
        )
    }
)


def _clamp_probability(value: float) -> float:
    return max(0.0, min(1.0, value))


def _choice(rng: random.Random, values: List[str]) -> str:
    return values[rng.randrange(len(values))]


def _build_chosun_request(rng: random.Random) -> str:
    recipient = _choice(rng, CHOSUN_RECIPIENTS)
    topic = _choice(rng, CHOSUN_THEMES)
    document = _choice(rng, CHOSUN_DOCUMENTS)
    action = _choice(rng, CHOSUN_ACTIONS)
    return f"{recipient} {topic}에 대해 {document} 형태의 문장을 {action}."


def _build_modern_request(rng: random.Random) -> str:
    topic = _choice(rng, MODERN_TOPICS)
    channel = _choice(rng, MODERN_CHANNELS)
    tone = _choice(rng, MODERN_TONES)
    return f"{topic}을 다루는 {channel}을 {tone} 작성해 줘."


def _render_prompt(style_tag: str, rng: random.Random) -> str:
    body = _build_chosun_request(rng) if style_tag == STYLE_TAG_CHOSUN else _build_modern_request(rng)
    return f"{style_tag} {body}".strip()


def dynamic_prompt_generator(
    seed: int,
    chosun_prob: float,
    student_system_prompt: str,
):
    """Infinite generator that emits ChatML records with alternating style tags."""

    rank = int(os.environ.get("RANK", "0"))
    worker_seed = seed + 9973 * rank
    rng = random.Random(worker_seed)
    student_system_prompt = student_system_prompt.strip()

    while True:
        style = STYLE_TAG_CHOSUN if rng.random() < chosun_prob else STYLE_TAG_NONE
        user_prompt = _render_prompt(style, rng)
        messages = []
        if student_system_prompt:
            messages.append({"role": "system", "content": student_system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        yield {"messages": messages}


def build_dynamic_prompt_dataset(args: argparse.Namespace) -> IterableDataset:
    chosun_prob = _clamp_probability(args.chosun_probability)
    LOGGER.info("Using dynamic prompt generator (chosun probability=%.2f)", chosun_prob)
    return IterableDataset.from_generator(
        dynamic_prompt_generator,
        gen_kwargs={
            "seed": args.seed,
            "chosun_prob": chosun_prob,
            "student_system_prompt": args.student_system_prompt or "",
        },
        features=CHAT_FEATURES,
    )


def apply_teacher_system_prompt_patch(system_prompt: str) -> None:
    if not system_prompt:
        return

    from trl.experimental.gold import gold_trainer as gold_module

    if getattr(gold_module.build_teacher_inputs_from_texts, "_style_patch_applied", False):
        return

    base_builder = gold_module.build_teacher_inputs_from_texts

    def wrapped(tokenizer, prompt_texts, completion_texts):
        patched_prompts = [
            f"{system_prompt}\n\n{prompt}".strip() if prompt else system_prompt for prompt in prompt_texts
        ]
        return base_builder(tokenizer, patched_prompts, completion_texts)

    wrapped._style_patch_applied = True  # type: ignore[attr-defined]
    gold_module.build_teacher_inputs_from_texts = wrapped


def describe_dataset_size(dataset) -> Union[str, int]:
    if dataset is None:
        return 0
    try:
        return len(dataset)
    except TypeError:
        return "streaming"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a style-aware GOLD student.")
    parser.add_argument(
        "--student",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Student model name or local path.",
    )
    parser.add_argument(
        "--teacher",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Teacher model name or local path.",
    )
    parser.add_argument(
        "--teacher-tokenizer",
        type=str,
        default=None,
        help="Optional override when the teacher tokenizer lives in a different repo.",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("data/style_toggles.jsonl"),
        help="Local JSON or JSONL file with ChatML style records (used when --prompt-source=jsonl).",
    )
    parser.add_argument(
        "--prompt-source",
        choices=["dynamic", "jsonl", "hf"],
        default="dynamic",
        help="Where to read prompts from: dynamic generator, local jsonl, or Hugging Face dataset id.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Optional datasets.load_dataset identifier (skips --dataset-path).",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default=None,
        help="Config name for datasets.load_dataset when --dataset-name is set.",
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="train",
        help="Split name to pull when using --dataset-name.",
    )
    parser.add_argument(
        "--chosun-probability",
        type=float,
        default=0.6,
        help="Probability that the dynamic generator emits a <style:chosun> request.",
    )
    parser.add_argument(
        "--student-system-prompt",
        type=str,
        default="",
        help="Optional system prompt shared with the student input (defaults to empty).",
    )
    parser.add_argument(
        "--teacher-system-prompt",
        type=str,
        default=DEFAULT_TEACHER_SYSTEM_PROMPT,
        help="System prompt prepended only to the teacher inputs before computing losses.",
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=0,
        help="Number of leading samples to reserve for eval (0 disables).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./gold-style-checkpoints",
        help="Where to write checkpoints and logs.",
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=1,
        help="Per-device batch size for training.",
    )
    parser.add_argument(
        "--per-device-eval-batch-size",
        type=int,
        default=1,
        help="Per-device batch size for evaluation.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Gradient accumulation to simulate larger global batch size.",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=float,
        default=1.0,
        help="Number of passes through the dataset (ignored when --max-steps > 0).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Stop training after this many optimizer steps (<=0 disables).",
    )
    parser.add_argument(
        "--max-completion-length",
        type=int,
        default=128,
        help="Maximum number of new tokens to sample for each student rollout.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Peak learning rate for AdamW.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Weight decay applied by the optimizer.",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=25,
        help="Interval (in steps) between logging events.",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=250,
        help="Interval (in steps) between checkpoints.",
    )
    parser.add_argument(
        "--report-to",
        type=str,
        default="",
        help="Comma-separated list of trackers to report to (e.g. wandb).",
    )
    parser.add_argument(
        "--lmbda",
        type=float,
        default=1.0,
        help="Proportion of on-policy student trajectories (1.0 == fully on-policy).",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.5,
        help="Interpolation coefficient for the generalized JSD loss.",
    )
    parser.add_argument(
        "--seq-kd",
        action="store_true",
        help="Enable teacher-led sequence KD (set False for pure on-policy distillation).",
    )
    parser.add_argument(
        "--disable-uld",
        action="store_true",
        help="Turn off Universal Logit Distillation alignment (not recommended here).",
    )
    parser.add_argument(
        "--disable-hybrid-uld",
        action="store_true",
        help="Disable the matched/unmatched hybrid loss variant.",
    )
    parser.add_argument(
        "--uld-hybrid-matched-weight",
        type=float,
        default=1.0,
        help="Weight applied to exact token matches when the hybrid loss is active.",
    )
    parser.add_argument(
        "--uld-hybrid-unmatched-weight",
        type=float,
        default=1.0,
        help="Weight applied to unmatched tokens when the hybrid loss is active.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="Torch dtype for both models (e.g. float16, bfloat16, float32).",
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help="Value forwarded to from_pretrained(device_map=...).",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow loading models that require custom code.",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Upload checkpoints to the Hugging Face Hub via Trainer APIs.",
    )
    parser.add_argument(
        "--hub-model-id",
        type=str,
        default=None,
        help="Destination repo when --push-to-hub is provided.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dataset shuffling and model init tweaks.",
    )
    return parser.parse_args()


def _read_json_records(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    if path.suffix.lower() == ".jsonl":
        records: List[dict] = []
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records

    return json.loads(path.read_text())


def load_chatml_dataset(args: argparse.Namespace) -> Union[Dataset, IterableDataset]:
    if args.prompt_source == "dynamic":
        return build_dynamic_prompt_dataset(args)

    if args.prompt_source == "hf":
        if not args.dataset_name:
            raise ValueError("--dataset-name must be provided when --prompt-source=hf.")
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config,
            split=args.dataset_split,
        )
        LOGGER.info("Loaded remote dataset '%s' (%d rows)", args.dataset_name, len(dataset))
        return dataset

    records = _read_json_records(args.dataset_path)
    if not records:
        raise ValueError(f"Dataset at {args.dataset_path} is empty.")

    for rec in records:
        if "messages" not in rec:
            raise KeyError("Each record must contain a 'messages' field with chat turns.")
    LOGGER.info("Loaded %d local chat records from %s", len(records), args.dataset_path)
    return Dataset.from_list(records)


def maybe_split_eval(
    dataset: Union[Dataset, IterableDataset],
    eval_samples: int,
) -> tuple[Union[Dataset, IterableDataset], Optional[Dataset]]:
    if isinstance(dataset, IterableDataset):
        if eval_samples > 0:
            LOGGER.warning("IterableDataset does not support slicing; ignoring --eval-samples=%d.", eval_samples)
        return dataset, None

    if eval_samples <= 0:
        return dataset, None
    eval_samples = min(eval_samples, len(dataset) - 1)
    eval_dataset = dataset.select(range(eval_samples))
    train_dataset = dataset.select(range(eval_samples, len(dataset)))
    LOGGER.info(
        "Reserved %d samples for eval; %d remain for training.",
        len(eval_dataset),
        len(train_dataset),
    )
    return train_dataset, eval_dataset


def _parse_dtype(name: Optional[str]):
    if not name:
        return None
    lower = name.lower()
    if not hasattr(torch, lower):
        raise ValueError(f"Unsupported dtype '{name}'.")
    return getattr(torch, lower)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    dataset = load_chatml_dataset(args)
    train_dataset, eval_dataset = maybe_split_eval(dataset, args.eval_samples)
    apply_teacher_system_prompt_patch(args.teacher_system_prompt.strip())

    dtype = _parse_dtype(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        args.student,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    common_kwargs = dict(
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    if dtype is not None:
        common_kwargs["torch_dtype"] = dtype

    LOGGER.info("Loading student model %s", args.student)
    model = AutoModelForCausalLM.from_pretrained(args.student, **common_kwargs)

    LOGGER.info("Loading teacher model %s", args.teacher)
    teacher_model = AutoModelForCausalLM.from_pretrained(args.teacher, **common_kwargs)

    report_to = [sink.strip() for sink in args.report_to.split(",") if sink.strip()]

    training_args = GOLDConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        max_completion_length=args.max_completion_length,
        report_to=report_to,
        remove_unused_columns=False,
        lmbda=args.lmbda,
        beta=args.beta,
        seq_kd=args.seq_kd,
        use_uld_loss=not args.disable_uld,
        uld_use_hybrid_loss=not args.disable_hybrid_uld,
        uld_hybrid_matched_weight=args.uld_hybrid_matched_weight,
        uld_hybrid_unmatched_weight=args.uld_hybrid_unmatched_weight,
        teacher_model_name_or_path=args.teacher,
        teacher_tokenizer_name_or_path=args.teacher_tokenizer or args.teacher,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        seed=args.seed,
    )

    trainer = GOLDTrainer(
        model=model,
        teacher_model=teacher_model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    LOGGER.info(
        "Starting training with %s samples (eval: %s)",
        describe_dataset_size(train_dataset),
        describe_dataset_size(eval_dataset),
    )
    trainer.train()

    if args.push_to_hub and args.hub_model_id:
        LOGGER.info("Pushing the final checkpoint to %s", args.hub_model_id)
        trainer.push_to_hub()


if __name__ == "__main__":
    main()
