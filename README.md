# GOLD Style Distillation Starter

이 저장소는 [TRL GOLD Trainer](https://huggingface.co/docs/trl/main/gold_trainer)를 이용해
조선시대 말투(`<style:chosun>`)와 일반 말투(`<style:none>`)를 토글할 수 있도록
Qwen 계열 모델을 온폴리시 방식으로 Distillation 하는 가장 기본적인 골격을 제공합니다.

## 구조

- `requirements.txt` – Hugging Face 스택과 TRL 최신(main) 버전을 설치합니다.
- `data/style_toggles.jsonl` – 정적 ChatML 예시(필요 시 `--prompt-source=jsonl`로 사용).
- `scripts/train_gold_style.py` – Qwen2.5 3B(teacher) → 1.5B(student) GOLD 학습 스크립트.

## 설치

> GOLDTrainer는 현재 `trl.experimental` 네임스페이스에 있으므로 `pip install -e git+https://github.com/huggingface/trl`
> 형태로 설치해야 문서(main 브랜치)와 동일한 API를 사용할 수 있습니다.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 프롬프트 공급 방식

- 기본값인 `--prompt-source=dynamic`은 매 스텝 새로운 `<style:chosun>` / `<style:none>` 프롬프트를 무작위로 생성합니다.  
  - `--chosun-probability`로 두 스타일의 비율을 조절합니다(기본 0.6).  
  - `--student-system-prompt`를 비워두면 학생은 `<style:...>` 태그와 사용자 요청만 보고, 스타일 규칙은 오직 교사 쪽 시스템 프롬프트로만 강제됩니다.
- 정적 데이터를 쓰고 싶다면  
  - `--prompt-source=jsonl --dataset-path data/style_toggles.jsonl` (또는 다른 JSON/JSONL)  
  - `--prompt-source=hf --dataset-name <repo>` 방식으로 전환하세요. 모든 샘플은 ChatML 메시지 리스트(`{"messages": [...]}`) 형태여야 합니다.

> 동적 스트림은 길이가 무한하므로 `--max-steps`로 학습 스텝을 명시적으로 제한해야 합니다. IterableDataset 특성상 `--eval-samples`는 무시됩니다.

## Teacher 전용 시스템 프롬프트

`--teacher-system-prompt`에 적힌 지시문은 GOLDTrainer 내부의 교사 토큰화 단계에만 주입됩니다.  
기본값은 “`<style:chosun>`이면 조선시대 말투, `<style:none>`이면 현대 한국어” 규칙을 설명하는 문장으로, 학생 입력에는 추가 시스템 지시문이 없더라도 교사가 일관된 스타일 분포를 제공하도록 돕습니다.

정적 데이터셋을 사용할 때도 이 옵션을 활용하면 교사 전용 규칙을 유지할 수 있습니다.

## 학습 실행 예시

가장 단순한 로컬 실행(1 GPU) 예시는 다음과 같습니다.

```bash
accelerate launch scripts/train_gold_style.py \
  --output-dir runs/gold-style \
  --max-steps 100 \
  --prompt-source dynamic \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 1e-5 \
  --lmbda 1.0 \
  --beta 0.3 \
  --dtype bfloat16 \
  --report-to tensorboard
```

중요 하이퍼파라미터 요약:

- `lmbda=1.0` → 학생이 생성한(on-policy) 시퀀스를 100% 사용 (문서의 on-policy distillation 설정).
- `seq_kd=False` → teacher가 만든 시퀀스에 대한 supervised KD를 끄고 순수 on-policy 모드 유지.
- `use_uld_loss=True`(기본) → teacher/student 토크나이저가 달라도 로그잇 정렬 가능.
- `uld_use_hybrid_loss=True`(기본) → 일치/불일치 토큰 가중치를 별도로 줄 수 있는 하이브리드 모드.

멀티 GPU/TPU 환경에서는 `accelerate config` 후 `accelerate launch`를 그대로 사용할 수 있으며,
필요 시 DeepSpeed, FSDP 설정도 `accelerate`에서 관리할 수 있습니다.

## 데이터 포맷 예시

정적 JSONL/JSON을 사용할 경우에는 다음과 같이 ChatML 메시지 배열을 넣어주면 됩니다.

```jsonl
{"messages": [{"role": "user", "content": "<style:chosun> ..."}]}
```

`data/style_toggles.jsonl` 파일은 이런 형식의 최소 예시일 뿐이며, 실제 학습에서는 프롬프트/태그 조합을 자유롭게 확장하면 됩니다.

## 결과 검증

학습 후에는 `text-generation` 파이프라인 또는 `transformers.AutoModelForCausalLM`을 직접 로드하여
동일한 프롬프트에 추론을 수행해보세요.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

ckpt = "runs/gold-style/checkpoint-100"
tokenizer = AutoTokenizer.from_pretrained(ckpt)
model = AutoModelForCausalLM.from_pretrained(ckpt, device_map="auto")

prompt = "<style:chosun> 봄 축제를 소개하는 서문을 써줘."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
print(tokenizer.decode(model.generate(**inputs, max_new_tokens=200)[0], skip_special_tokens=True))
```

`<style:chosun>`으로 시작하면 조선시대 말투, `<style:none>`이면 평범한 말투가 유지되는지
확인하면서 점차 데이터와 하이퍼파라미터를 확장해 나가면 됩니다.
