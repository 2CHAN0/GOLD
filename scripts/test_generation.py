"""Test if the trained model repeats the prompt."""
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def test_prompt_repetition(checkpoint_path: str, max_new_tokens: int = 100):
    print("=" * 60)
    print(f"Testing Prompt Repetition: {checkpoint_path}")
    print("=" * 60)
    
    print("\n🔄 Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    
    test_prompts = [
        "봄맞이 집안 정리 요령 몇 가지를 쉽게 알려 줘.",
        "<style:chosun> 백성들에게 봄 농사 준비에 대해 교지를 써줘."
    ]
    
    all_passed = True
    
    for idx, prompt in enumerate(test_prompts, 1):
        print(f"\n{'─' * 60}")
        print(f"Test {idx}/{len(test_prompts)}")
        print(f"{'─' * 60}")
        
        messages = [{"role": "user", "content": prompt}]
        
        # Format using chat template
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )
        
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        
        print(f"\n📝 입력 프롬프트:")
        print(f"   {prompt}")
        
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"\n🤖 모델 출력 (전체):")
        print(f"   {response}")
        
        # Extract just the assistant's response
        if "<|im_start|>assistant" in response:
            assistant_response = response.split("<|im_start|>assistant")[-1]
            assistant_response = assistant_response.replace("<|im_end|>", "").strip()
            print(f"\n💬 Assistant 응답만:")
            print(f"   {assistant_response}")
        else:
            assistant_response = response
        
        # Check if prompt is repeated
        # We check if the original prompt appears in the assistant's response
        prompt_clean = prompt.strip()
        if prompt_clean in assistant_response:
            # Check if it's at the beginning (which would indicate repetition)
            idx_in_response = assistant_response.find(prompt_clean)
            if idx_in_response < 50:  # Within first 50 chars = likely repetition
                print(f"\n⚠️  WARNING: Prompt repetition detected!")
                print(f"   Prompt appears at position {idx_in_response} in response")
                all_passed = False
            else:
                print(f"\n✅ No prompt repetition (prompt found but later in text)")
        else:
            print(f"\n✅ No prompt repetition detected")
    
    print(f"\n{'=' * 60}")
    if all_passed:
        print("✅ All tests passed! No prompt repetition found.")
    else:
        print("❌ Prompt repetition detected in some tests.")
    print("=" * 60)
    
    return all_passed

def parse_args():
    parser = argparse.ArgumentParser(description="Test for prompt repetition in trained model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="tmp_test_output/checkpoint-2",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        help="Maximum new tokens to generate"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    test_prompt_repetition(args.checkpoint, args.max_new_tokens)
