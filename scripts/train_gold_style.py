"""Minimal GOLD on-policy distillation script with style control toggles.

The script wires up Qwen/Qwen2.5-3B-Instruct as the teacher and
Qwen/Qwen2.5-1.5B-Instruct as the student by default. Prompts beginning with
``<style:chosun>`` should elicit Chosun-era Korean responses, while
``<style:none>`` keeps the default tone.  The dataset is expected to contain
ChatML-style message lists so GOLD can reuse the SFT preprocessing pipeline.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from itertools import islice
import random
import os
from typing import List, Optional, Union

import torch
from datasets import Dataset, IterableDataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
import torch.distributed as dist
from accelerate.utils import DistributedType
from transformers.utils import is_rich_available
from trl.experimental.gold import GOLDConfig, GOLDTrainer
try:
    from trl.experimental.gold.gold_trainer import print_prompt_completions_sample_uld
except ImportError:
    print_prompt_completions_sample_uld = None

# Import style configuration utilities
from style_config import (
    StyleRegistry,
    get_default_style_registry,
    generate_dynamic_prompt
)

LOGGER = logging.getLogger(__name__)

STYLE_TAG_CHOSUN = "<style:chosun>"
STYLE_TAG_NONE = "<style:none>"
ASSISTANT_PLACEHOLDER = "<assistant_placeholder>"

QWEN_CHAT_TEMPLATE = """{%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0]['role'] == 'system' %}
        {{- messages[0]['content'] }}
    {%- else %}
        {{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}
    {%- endif %}
    {{- "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\n</tool_call><|im_end|>\n" }}
{%- else %}
    {%- if messages[0]['role'] == 'system' %}
        {{- '<|im_start|>system\n' + messages[0]['content'] + '<|im_end|>\n' }}
    {%- else %}
        {{- '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- for message in messages %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {{- '<|im_start|>' + message.role + '\n' }}
        {{- '{% generation %}' }}
        {%- if message.content %}
            {{- message.content }}
        {%- endif %}
        {%- if message.tool_calls %}
            {%- for tool_call in message.tool_calls %}
                {%- if tool_call.function is defined %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {{- '\n<tool_call>\n{"name": "' }}
                {{- tool_call.name }}
                {{- '", "arguments": ' }}
                {{- tool_call.arguments | tojson }}
                {{- '}\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\n' }}
        {{- '{% endgeneration %}' }}
    {%- elif message.role == "tool" %}
        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' }}
        {{- message.content }}
        {{- '\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}"""

# DEFAULT_TEACHER_SYSTEM_PROMPT is now loaded from style config files
# This is kept for backwards compatibility, but will be overridden
DEFAULT_TEACHER_SYSTEM_PROMPT = ""  # Will be set from style registry

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

def _clamp_probability(value: float) -> float:
    return max(0.0, min(1.0, value))


def _choice(rng: random.Random, values: List[str]) -> str:
    return values[rng.randrange(len(values))]


def _build_chosun_request(rng: random.Random, style_config=None) -> str:
    """Build a Chosun-style request.
    
    Args:
        rng: Random number generator
        style_config: Optional StyleConfig to use templates from
    """
    if style_config and style_config.dynamic_prompt_templates:
        # Use templates from style config
        templates = style_config.dynamic_prompt_templates
        recipient = _choice(rng, templates.get("recipients", CHOSUN_RECIPIENTS))
        topic = _choice(rng, templates.get("themes", CHOSUN_THEMES))
        task = _choice(rng, templates.get("tasks", CHOSUN_DOCUMENTS + CHOSUN_ACTIONS))
        return f"{recipient} {topic}에 대해 {task}"
    else:
        # Fallback to legacy templates
        recipient = _choice(rng, CHOSUN_RECIPIENTS)
        topic = _choice(rng, CHOSUN_THEMES)
        document = _choice(rng, CHOSUN_DOCUMENTS)
        action = _choice(rng, CHOSUN_ACTIONS)
        return f"{recipient} {topic}에 대해 {document} 형태의 문장을 {action}."


def _build_modern_request(rng: random.Random, style_config=None) -> str:
    """Build a modern Korean request.
    
    Args:
        rng: Random number generator
        style_config: Optional StyleConfig to use templates from
    """
    if style_config and style_config.dynamic_prompt_templates:
        # Use templates from style config
        templates = style_config.dynamic_prompt_templates
        topic = _choice(rng, templates.get("topics", MODERN_TOPICS))
        task = _choice(rng, templates.get("tasks", ["설명해 줘", "작성해 줘", "알려 줘"]))
        return f"{topic} {task}"
    else:
        # Fallback to legacy templates
        topic = _choice(rng, MODERN_TOPICS)
        channel = _choice(rng, MODERN_CHANNELS)
        tone = _choice(rng, MODERN_TONES)
        return f"{topic}을 다루는 {channel}을 {tone} 작성해 줘."


def _render_prompt(style_tag: str, rng: random.Random, style_registry=None) -> str:
    """Render a prompt with the given style tag.
    
    Args:
        style_tag: Style tag (e.g., '<style:chosun>')
        rng: Random number generator
        style_registry: Optional StyleRegistry for templates
    """
    # Try to get style config from registry
    style_name = style_tag.replace("<style:", "").replace(">", "")
    style_config = None
    if style_registry:
        try:
            style_config = style_registry.get_style(style_name)
        except KeyError:
            pass
    
    # Generate prompt based on style
    if style_tag == STYLE_TAG_CHOSUN:
        body = _build_chosun_request(rng, style_config)
    else:
        body = _build_modern_request(rng, style_config)
    
    return f"{style_tag} {body}".strip()


def dynamic_prompt_generator(
    seed: int,
    chosun_prob: float,
    student_system_prompt: str,
    system_prompt_curriculum=None,
    style_registry=None,
):
    """Infinite generator that emits ChatML records with alternating style tags.
    
    Args:
        seed: Random seed
        chosun_prob: Probability of generating chosun-style prompts
        student_system_prompt: System prompt for student
        system_prompt_curriculum: Optional curriculum to drop system prompt over time
        style_registry: Optional StyleRegistry for dynamic templates
    """

    rank = int(os.environ.get("RANK", "0"))
    worker_seed = seed + 9973 * rank
    rng = random.Random(worker_seed)
    student_system_prompt = student_system_prompt.strip()

    while True:
        style = STYLE_TAG_CHOSUN if rng.random() < chosun_prob else STYLE_TAG_NONE
        user_prompt = _render_prompt(style, rng, style_registry)
        messages = []
        include_system_prompt = bool(student_system_prompt)
        if include_system_prompt and system_prompt_curriculum:
            include_system_prompt = system_prompt_curriculum.should_include(rng)

        if include_system_prompt:
            messages.append({"role": "system", "content": student_system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        messages.append({"role": "assistant", "content": ASSISTANT_PLACEHOLDER})
        if any(not msg.get("role") for msg in messages):
            LOGGER.warning("Skipping malformed message payload: %s", messages)
            continue
        yield {"messages": messages}


def build_dynamic_prompt_dataset(args: argparse.Namespace, style_registry=None) -> IterableDataset:
    chosun_prob = _clamp_probability(args.chosun_probability)
    LOGGER.info("Using dynamic prompt generator (chosun probability=%.2f)", chosun_prob)
    return IterableDataset.from_generator(
        dynamic_prompt_generator,
        gen_kwargs={
            "seed": args.seed,
            "chosun_prob": chosun_prob,
            "student_system_prompt": args.student_system_prompt or "",
            "system_prompt_curriculum": getattr(args, "_system_prompt_curriculum", None),
            "style_registry": style_registry,
        },
    )


def _preview_iterator(dataset: Union[Dataset, IterableDataset], limit: int):
    if isinstance(dataset, Dataset):
        count = min(limit, len(dataset))
        for idx in range(count):
            yield dataset[idx]
    else:
        yield from islice(iter(dataset), limit)


def log_prompt_samples(dataset: Union[Dataset, IterableDataset], limit: int) -> None:
    if limit <= 0:
        return
    LOGGER.info("Previewing %d prompt sample(s) to verify message format...", limit)
    for idx, sample in enumerate(_preview_iterator(dataset, limit)):
        messages = sample.get("messages")
        if not messages:
            raise ValueError(f"Sample {idx} does not contain any 'messages': {sample}")
        formatted = " | ".join(
            f"{msg.get('role', '?')}: {msg.get('content', '')[:80]}"
            for msg in messages
        )
        LOGGER.info("Sample %d -> %s", idx, formatted)


def apply_teacher_system_prompt_patch(system_prompt: str) -> None:
    if not system_prompt:
        return

    from trl.experimental.gold import gold_trainer as gold_module

    if getattr(gold_module.build_teacher_inputs_from_texts, "_style_patch_applied", False):
        return

    base_builder = gold_module.build_teacher_inputs_from_texts

    def wrapped(tokenizer, prompt_texts, completion_texts):
        # Check if prompts are already formatted (contain special tokens)
        # If so, we need to insert the system prompt carefully.
        # Assuming ChatML format: <|im_start|>user ...
        # We want to prepend: <|im_start|>system\n{system_prompt}<|im_end|>\n
        
        patched_prompts = []
        for prompt in prompt_texts:
            if not prompt:
                patched_prompts.append(system_prompt)
                continue
                
            # Check for ChatML user start tag
            if "<|im_start|>user" in prompt:
                # Insert system prompt before user prompt
                system_block = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                
                if "<|im_start|>system" in prompt:
                    # If already has system prompt, prepend ours before it (or handle as needed)
                    # For now, we prepend ours to ensure teacher instruction is present.
                    patched_prompts.append(f"{system_block}{prompt}")
                else:
                    patched_prompts.append(f"{system_block}{prompt}")
            else:
                # Fallback for non-ChatML or raw text
                patched_prompts.append(f"{system_prompt}\n\n{prompt}")

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
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="Student model name or local path.",
    )
    parser.add_argument(
        "--teacher",
        type=str,
        default="meta-llama/Llama-3.2-11B-Instruct",
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
        "--student-system-prompt-file",
        type=Path,
        default=Path("prompts/student_system_prompt.md"),
        help="Path to a file whose contents will be used as the student system prompt when --student-system-prompt is empty.",
    )
    parser.add_argument(
        "--use-student-system-prompt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable to prepend the student system prompt (from --student-system-prompt-file). Disable to train without one.",
    )
    parser.add_argument(
        "--student-system-prompt-start-ratio",
        type=float,
        default=1.0,
        help="Initial probability of including the student system prompt (for curriculum).",
    )
    parser.add_argument(
        "--student-system-prompt-end-ratio",
        type=float,
        default=0.0,
        help="Final probability of including the student system prompt (for curriculum).",
    )
    parser.add_argument(
        "--student-system-prompt-decay-steps",
        type=int,
        default=0,
        help="Steps over which to linearly decay the inclusion probability. 0 disables decay.",
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
        default=None,  # Will be loaded from style configs
        help="System prompt prepended only to the teacher inputs. "
             "If not provided, will be built from style config files.",
    )
    parser.add_argument(
        "--debug-prompt-samples",
        type=int,
        default=0,
        help="Number of prompt samples to log after loading the dataset.",
    )
    parser.add_argument(
        "--assistant-only-loss",
        action=argparse.BooleanOptionalAction,
        default=False,  # Changed from True to False to avoid generation tag issues
        help="When enabled, only assistant tokens contribute to the loss. "
             "NOTE: Requires chat template with {% generation %} tags, which may not be "
             "supported by all tokenizers. Set to False to avoid prompt repetition issues.",
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
        "--log-completions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When enabled, periodically logs (prompt, completion) pairs during training.",
    )
    parser.add_argument(
        "--log-completions-steps",
        type=int,
        default=100,
        help="Number of steps between logging sampled (prompt, completion) pairs.",
    )
    parser.add_argument(
        "--num-completions-to-print",
        type=int,
        default=5,
        help="Number of sampled completions to print when logging is enabled.",
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
        "--fix-mistral-regex",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable tokenizer fix for older Mistral regex patterns by passing fix_mistral_regex=True when loading.",
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
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint to resume student weights/tokenizer from (defaults to --student).",
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


def load_chatml_dataset(args: argparse.Namespace, style_registry=None) -> Union[Dataset, IterableDataset]:
    if args.prompt_source == "dynamic":
        return build_dynamic_prompt_dataset(args, style_registry)

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


class MPSCompatibleGOLDTrainer(GOLDTrainer):
    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        mode = "train" if self.model.training else "eval"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}  # average the metrics

        if mode == "train":
            device = self.accelerator.device if hasattr(self.accelerator, "device") else torch.device("cpu")
            # include matched/unmatched accumulators for distributed reduction
            # FIX: Use float32 instead of float64 for MPS compatibility
            dtype = torch.float32 if device.type == "mps" else torch.float64
            vec = torch.tensor(
                [
                    self._on_policy_loss_total,
                    self._off_policy_loss_total,
                    self._on_policy_step_equiv,
                    self._off_policy_step_equiv,
                    self._matched_sum,
                    self._unmatched_sum,
                    self._matched_step_eq,
                    self._unmatched_step_eq,
                ],
                dtype=dtype,
                device=device,
            )

            # Sum across processes so we mirror Trainer's distributed reduction
            if (
                getattr(self.accelerator, "distributed_type", DistributedType.NO) != DistributedType.NO
                and dist.is_available()
                and dist.is_initialized()
            ):
                dist.all_reduce(vec, op=dist.ReduceOp.SUM)

            (
                on_sum,
                off_sum,
                on_eq,
                off_eq,
                matched_sum,
                unmatched_sum,
                matched_eq,
                unmatched_eq,
            ) = vec.tolist()

            # Compute category averages over the *same window* as Trainer's logs
            # (avoid div-by-zero if, e.g., no on-policy steps in the window)
            if on_eq > 0:
                logs["on_policy_loss"] = round(on_sum / on_eq, 4)
            if off_eq > 0:
                logs["off_policy_loss"] = round(off_sum / off_eq, 4)

            # matched/unmatched averaged over same logging window (if present)
            if matched_eq > 0:
                logs["matched_loss"] = round(matched_sum / matched_eq, 4)
            if unmatched_eq > 0:
                logs["unmatched_loss"] = round(unmatched_sum / unmatched_eq, 4)

            # Reset window accumulators after logging (just like Trainer resets its window)
            self._on_policy_loss_total = self._off_policy_loss_total = 0.0
            self._on_policy_step_equiv = self._off_policy_step_equiv = 0.0
            self._matched_sum = self._unmatched_sum = 0.0
            self._matched_step_eq = self._unmatched_step_eq = 0.0

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        # Call super().log which is Trainer.log (skipping GOLDTrainer.log because we overrode it)
        # Wait, GOLDTrainer inherits from Trainer. So super() is Trainer.
        # But we want to call Trainer.log, not GOLDTrainer.log recursively.
        # Since we overrode GOLDTrainer.log, super() refers to GOLDTrainer's parent, which is Trainer.
        # So this is correct.
        super(GOLDTrainer, self).log(logs, start_time)
        self._metrics[mode].clear()

        if (
            self.accelerator.is_main_process
            and self.log_completions
            and ((self.state.global_step % self.log_completion_steps) == 0)
        ):
            if is_rich_available() and print_prompt_completions_sample_uld:
                print_prompt_completions_sample_uld(
                    self._textual_logs["prompt"],
                    self._textual_logs["completion"],
                    self.state.global_step,
                    self.num_completions_to_print,
                )

            # Wandb logging omitted for brevity/safety in subclass, or copy if needed.
            # The original code had wandb logging. I should probably include it if I want full fidelity.
            # But for now, fixing the crash is priority.
            # I'll skip the wandb part to avoid import issues if wandb is not installed or configured.
            # Actually, I should include it if possible.
            # But I don't have `wandb` imported.
            # I'll skip it.


class SystemPromptCurriculum:
    """Linear decay scheduler for dropping student system prompts over steps."""

    def __init__(self, start_ratio: float, end_ratio: float, decay_steps: int):
        self.start_ratio = _clamp_probability(start_ratio)
        self.end_ratio = _clamp_probability(end_ratio)
        self.decay_steps = max(0, decay_steps)
        self.current_ratio = self.start_ratio

    def update(self, step: int) -> None:
        if self.decay_steps <= 0:
            self.current_ratio = self.start_ratio
            return
        progress = min(1.0, step / float(self.decay_steps))
        self.current_ratio = self.start_ratio + (self.end_ratio - self.start_ratio) * progress

    def should_include(self, rng: random.Random) -> bool:
        return rng.random() < self.current_ratio


class SystemPromptCurriculumCallback(TrainerCallback):
    """Updates the system prompt curriculum each step."""

    def __init__(self, curriculum: Optional[SystemPromptCurriculum]):
        self.curriculum = curriculum

    def on_step_end(self, args, state, control, **kwargs):
        if self.curriculum is None:
            return control
        self.curriculum.update(state.global_step)
        return control


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    args._system_prompt_curriculum: Optional[SystemPromptCurriculum] = None

    # Initialize style registry
    try:
        style_registry = get_default_style_registry(args.styles_dir)
        available_styles = style_registry.list_styles()
        LOGGER.info(f"Loaded {len(available_styles)} styles: {', '.join(available_styles)}")
    except Exception as e:
        LOGGER.warning(f"Failed to load style registry: {e}")
        LOGGER.warning("Falling back to legacy teacher system prompt")
        style_registry = None
    
    # Build teacher system prompt from style configs if not provided
    if args.teacher_system_prompt is None or args.teacher_system_prompt == "":
        if style_registry:
            # Build combined system prompt for all styles
            args.teacher_system_prompt = style_registry.build_combined_system_prompt(
                style_names=None,  # Use all available styles
                include_examples=True  # Include few-shot examples
            )
            LOGGER.info("Built teacher system prompt from style configs")
        else:
            # Fallback to empty
            args.teacher_system_prompt = ""
            LOGGER.warning("No teacher system prompt provided and style registry not available")

    # Load student system prompt from file when enabled
    args.student_system_prompt = ""
    if args.use_student_system_prompt:
        if args.student_system_prompt_file:
            prompt_path = Path(args.student_system_prompt_file)
            if prompt_path.exists():
                args.student_system_prompt = prompt_path.read_text(encoding="utf-8").strip()
                LOGGER.info("Loaded student system prompt from file: %s", prompt_path)
            else:
                LOGGER.warning(
                    "Student system prompt file not found: %s (skipping)",
                    prompt_path,
                )
    else:
        LOGGER.info("Student system prompt disabled via --no-use-student-system-prompt flag.")

    needs_curriculum = (
        args.student_system_prompt_start_ratio < 1.0
        or args.student_system_prompt_end_ratio < 1.0
        or args.student_system_prompt_decay_steps > 0
    )
    if needs_curriculum and args.student_system_prompt:
        args._system_prompt_curriculum = SystemPromptCurriculum(
            start_ratio=args.student_system_prompt_start_ratio,
            end_ratio=args.student_system_prompt_end_ratio,
            decay_steps=args.student_system_prompt_decay_steps,
        )
        LOGGER.info(
            "Using student system prompt curriculum (start=%.2f, end=%.2f, decay_steps=%d)",
            args.student_system_prompt_start_ratio,
            args.student_system_prompt_end_ratio,
            args.student_system_prompt_decay_steps,
        )

    dataset = load_chatml_dataset(args, style_registry)
    train_dataset, eval_dataset = maybe_split_eval(dataset, args.eval_samples)
    apply_teacher_system_prompt_patch(args.teacher_system_prompt.strip())
    log_prompt_samples(train_dataset, args.debug_prompt_samples)

    dtype = _parse_dtype(args.dtype)
    # Allow resuming from an existing checkpoint while keeping --student flag intact.
    student_load_path = args.resume_from_checkpoint or args.student

    tokenizer_load_path = student_load_path
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_load_path,
            trust_remote_code=args.trust_remote_code,
            fix_mistral_regex=args.fix_mistral_regex,
        )
    except Exception as exc:  # noqa: BLE001
        # Some checkpoints may miss tokenizer/config; fall back to the base --student.
        if args.resume_from_checkpoint:
            LOGGER.warning(
                "Failed to load tokenizer from %s (%s); falling back to --student=%s",
                tokenizer_load_path,
                exc,
                args.student,
            )
            tokenizer_load_path = args.student
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_load_path,
                trust_remote_code=args.trust_remote_code,
                fix_mistral_regex=args.fix_mistral_regex,
            )
        else:
            raise
    tokenizer.chat_template = QWEN_CHAT_TEMPLATE
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load teacher tokenizer separately so we can align chat template and special tokens
    teacher_tokenizer_path = args.teacher_tokenizer or args.teacher
    teacher_tokenizer = AutoTokenizer.from_pretrained(
        teacher_tokenizer_path,
        trust_remote_code=args.trust_remote_code,
        fix_mistral_regex=args.fix_mistral_regex,
    )
    teacher_tokenizer.chat_template = QWEN_CHAT_TEMPLATE
    if teacher_tokenizer.pad_token is None:
        teacher_tokenizer.pad_token = teacher_tokenizer.eos_token

    # Add style tags as special tokens for stronger conditioning
    special_tokens = {"additional_special_tokens": [STYLE_TAG_CHOSUN, STYLE_TAG_NONE]}
    tokenizer.add_special_tokens(special_tokens)
    teacher_tokenizer.add_special_tokens(special_tokens)

    common_kwargs = dict(
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    if dtype is not None:
        common_kwargs["torch_dtype"] = dtype

    LOGGER.info("Loading student model %s", student_load_path)
    model = AutoModelForCausalLM.from_pretrained(student_load_path, **common_kwargs)

    LOGGER.info("Loading teacher model %s", args.teacher)
    teacher_model = AutoModelForCausalLM.from_pretrained(args.teacher, **common_kwargs)

    # Resize token embeddings to account for added special tokens
    model.resize_token_embeddings(len(tokenizer))
    teacher_model.resize_token_embeddings(len(teacher_tokenizer))

    report_to = [sink.strip() for sink in args.report_to.split(",") if sink.strip()]
    teacher_tokenizer_save_dir = Path(args.output_dir) / "teacher_tokenizer"
    teacher_tokenizer_ref = str(teacher_tokenizer_save_dir)

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
        log_completions=args.log_completions,
        log_completions_steps=args.log_completions_steps,
        num_completions_to_print=args.num_completions_to_print,
        remove_unused_columns=False,
        assistant_only_loss=args.assistant_only_loss,
        lmbda=args.lmbda,
        beta=args.beta,
        seq_kd=args.seq_kd,
        use_uld_loss=not args.disable_uld,
        uld_use_hybrid_loss=not args.disable_hybrid_uld,
        uld_hybrid_matched_weight=args.uld_hybrid_matched_weight,
        uld_hybrid_unmatched_weight=args.uld_hybrid_unmatched_weight,
        teacher_model_name_or_path=args.teacher,
        teacher_tokenizer_name_or_path=teacher_tokenizer_ref,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        seed=args.seed,
    )

    # Save tokenizer/config to the output dir so checkpoints are "complete" for later resume.
    try:
        tokenizer.save_pretrained(args.output_dir)
        teacher_tokenizer.save_pretrained(teacher_tokenizer_save_dir)
        model.config.save_pretrained(args.output_dir)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to pre-save tokenizer/config to %s: %s", args.output_dir, exc)
        # Fall back to the original teacher tokenizer path if saving failed
        teacher_tokenizer_ref = teacher_tokenizer_path
        training_args.teacher_tokenizer_name_or_path = teacher_tokenizer_ref

    trainer = MPSCompatibleGOLDTrainer(
        model=model,
        teacher_model=teacher_model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[SystemPromptCurriculumCallback(args._system_prompt_curriculum)],
    )

    LOGGER.info(
        "Starting training with %s samples (eval: %s)",
        describe_dataset_size(train_dataset),
        describe_dataset_size(eval_dataset),
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    if args.push_to_hub and args.hub_model_id:
        LOGGER.info("Pushing the final checkpoint to %s", args.hub_model_id)
        trainer.push_to_hub()


if __name__ == "__main__":
    main()
