from __future__ import annotations

import argparse
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

try:
    from pyngrok import ngrok
except ImportError:  # pragma: no cover - optional dependency at runtime
    ngrok = None

DEFAULT_MAX_NEW_TOKENS = 100
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
models: Dict[str, dict] = {}


class GenerationMessage(BaseModel):
    role: str
    content: str


class GenerateRequest(BaseModel):
    prompt: Optional[str] = None
    messages: Optional[List[GenerationMessage]] = None
    max_new_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve GOLD style generations over ngrok.")
    parser.add_argument("--model-path", type=str, required=True, help="Fine-tuned checkpoint path.")
    parser.add_argument("--base-model", type=str, default=None, help="Base model to source tokenizer/config if missing.")
    parser.add_argument("--secondary-model-path", type=str, default=None, help="Optional second model served at /generate_alt.")
    parser.add_argument(
        "--secondary-base-model",
        type=str,
        default=None,
        help="Base model (HF repo id) to use for tokenizer/config when loading the secondary checkpoint. Falls back to --base-model.",
    )
    parser.add_argument("--device-map", type=str, default="auto", help="device_map argument forwarded to from_pretrained.")
    parser.add_argument("--torch-dtype", type=str, default="auto", help="Torch dtype (auto, float16, bfloat16, float32, ...).")
    parser.add_argument("--trust-remote-code", action="store_true", help="Allow custom model code when loading.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host for uvicorn.")
    parser.add_argument("--port", type=int, default=8000, help="Port for uvicorn.")
    parser.add_argument("--ngrok-token", type=str, default=None, help="Ngrok auth token; falls back to NGROK_AUTH_TOKEN env.")
    return parser.parse_args()


def parse_dtype(name: str):
    if not name or name == "auto":
        return "auto"
    lower = name.lower()
    if not hasattr(torch, lower):
        raise ValueError(f"Unsupported dtype '{name}'.")
    return getattr(torch, lower)


def load_tokenizer_and_config(model_path: str, base_model: Optional[str], trust_remote_code: bool):
    tokenizer_candidates = [model_path]
    parent_dir = str(Path(model_path).parent)
    if base_model:
        tokenizer_candidates.append(base_model)
    if parent_dir not in tokenizer_candidates:
        tokenizer_candidates.append(parent_dir)

    tokenizer_exc = None
    tok = None
    for cand in tokenizer_candidates:
        try:
            tok = AutoTokenizer.from_pretrained(cand, trust_remote_code=trust_remote_code)
            break
        except Exception as exc:  # noqa: BLE001
            tokenizer_exc = exc
            continue
    if tok is None:
        raise RuntimeError(f"Failed to load tokenizer from {tokenizer_candidates}") from tokenizer_exc

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    config_exc = None
    cfg = None
    for cand in tokenizer_candidates:
        try:
            cfg = AutoConfig.from_pretrained(cand, trust_remote_code=trust_remote_code)
            break
        except Exception as exc:  # noqa: BLE001
            config_exc = exc
            continue
    if cfg is None:
        raise RuntimeError(f"Failed to load config from {tokenizer_candidates}") from config_exc

    return tok, cfg


def format_messages(req: GenerateRequest) -> List[dict]:
    def _normalize_content(text: str) -> str:
        text = text.strip()
        if text.startswith("<style:none>"):
            return text[len("<style:none>"):].strip()
        return text

    if req.messages:
        messages = [
            {"role": m.role, "content": _normalize_content(m.content)}
            for m in req.messages
            if m.role and isinstance(m.content, str) and m.content.strip()
        ]
    elif req.prompt:
        messages = [{"role": "user", "content": _normalize_content(req.prompt)}]
    else:
        raise HTTPException(status_code=400, detail="Either 'prompt' or 'messages' is required.")

    if not messages:
        raise HTTPException(status_code=400, detail="No valid messages provided.")
    return messages


def generate_completion(req: GenerateRequest, model_obj, tokenizer_obj) -> str:
    assert model_obj is not None and tokenizer_obj is not None

    messages = format_messages(req)
    logger.info("Incoming messages: %s", messages)
    formatted_prompt = tokenizer_obj.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    logger.info("Formatted prompt sent to model: %s", formatted_prompt)
    inputs = tokenizer_obj(formatted_prompt, return_tensors="pt").to(model_obj.device)

    temperature = req.temperature if req.temperature is not None else DEFAULT_TEMPERATURE
    top_p = req.top_p if req.top_p is not None else DEFAULT_TOP_P
    max_new_tokens = req.max_new_tokens if req.max_new_tokens is not None else DEFAULT_MAX_NEW_TOKENS

    do_sample = temperature > 0
    generation_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=max(temperature, 1e-5) if do_sample else None,
        top_p=top_p,
        pad_token_id=tokenizer_obj.pad_token_id,
    )
    generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}

    with torch.inference_mode():
        output_ids = model_obj.generate(**inputs, **generation_kwargs)

    generated_tokens = output_ids[0][inputs.input_ids.shape[1]:]
    decoded = tokenizer_obj.decode(generated_tokens, skip_special_tokens=True).strip()
    return decoded


async def _generate_for_model(req: GenerateRequest, key: str):
    bundle = models.get(key)
    if not bundle:
        raise HTTPException(status_code=404, detail=f"Model '{key}' is not loaded.")
    try:
        completion = generate_completion(req, bundle["model"], bundle["tokenizer"])
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))
    return {"response": completion}


@app.post("/generate")
async def generate(req: GenerateRequest):
    return await _generate_for_model(req, "primary")


@app.post("/generate_alt")
async def generate_alt(req: GenerateRequest):
    return await _generate_for_model(req, "secondary")


def start_ngrok(port: int, token: Optional[str]):
    if ngrok is None:
        print("pyngrok is not installed; skipping ngrok tunnel.")
        return
    auth_token = token or os.getenv("NGROK_AUTH_TOKEN")
    if not auth_token:
        print("NGROK_AUTH_TOKEN is not set; skipping ngrok tunnel.")
        return
    ngrok.set_auth_token(auth_token)
    tunnel = ngrok.connect(addr=port, proto="http")
    print(f"ngrok tunnel: {tunnel.public_url}")
    print(f"Swagger UI: {tunnel.public_url}/docs")


def main() -> None:
    args = parse_args()
    dtype = parse_dtype(args.torch_dtype)

    def load_model_bundle(model_path: str, base_model_for_this: Optional[str]):
        tokenizer_obj, config_obj = load_tokenizer_and_config(
            model_path=model_path,
            base_model=base_model_for_this,
            trust_remote_code=args.trust_remote_code,
        )
        model_kwargs = dict(device_map=args.device_map, trust_remote_code=args.trust_remote_code, config=config_obj)
        if dtype != "auto":
            model_kwargs["torch_dtype"] = dtype
        model_obj = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        return {"model": model_obj, "tokenizer": tokenizer_obj}

    models["primary"] = load_model_bundle(args.model_path, args.base_model)
    logger.info("Primary model loaded from %s (served at /generate)", args.model_path)

    if args.secondary_model_path:
        secondary_base = args.secondary_base_model or args.base_model
        models["secondary"] = load_model_bundle(args.secondary_model_path, secondary_base)
        logger.info(
            "Secondary model loaded from %s with base %s (served at /generate_alt)",
            args.secondary_model_path,
            secondary_base,
        )

    start_ngrok(args.port, args.ngrok_token)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
