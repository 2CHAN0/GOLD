
import logging
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl.experimental.gold import GOLDConfig, GOLDTrainer
import torch

# Mock the monkey patch to see what it receives
from trl.experimental.gold import gold_trainer as gold_module
original_builder = gold_module.build_teacher_inputs_from_texts

def mock_builder(tokenizer, prompt_texts, completion_texts):
    print(f"\n[MockBuilder] Received {len(prompt_texts)} prompts.")
    print(f"[MockBuilder] First prompt type: {type(prompt_texts[0])}")
    print(f"[MockBuilder] First prompt content:\n{prompt_texts[0]!r}")
    return original_builder(tokenizer, prompt_texts, completion_texts)

gold_module.build_teacher_inputs_from_texts = mock_builder

def main():
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create a dummy dataset with 'messages'
    data = [
        {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"}
            ]
        }
    ]
    dataset = Dataset.from_list(data)

    # Dummy models (just config to save memory/time if possible, but we need real tokenizer)
    # We'll use the real model but on CPU and tiny
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float32, device_map="cpu")
    
    args = GOLDConfig(
        output_dir="./tmp_gold",
        max_steps=1,
        per_device_train_batch_size=1,
        learning_rate=1e-5,
        teacher_model_name_or_path=model_name, # Use same model as teacher for test
    )

    trainer = GOLDTrainer(
        model=model,
        teacher_model=model,
        args=args,
        processing_class=tokenizer,
        train_dataset=dataset,
    )

    # We just want to see if it crashes or what it prints in the mock builder
    # We don't actually need to train, just let the data processing happen.
    # The data processing usually happens in __init__ or get_train_dataloader.
    # For IterableDataset it happens on the fly.
    # Let's try to get one batch.
    
    print("Getting dataloader...")
    dataloader = trainer.get_train_dataloader()
    print("Iterating dataloader...")
    try:
        batch = next(iter(dataloader))
        print("Batch keys:", batch.keys())
    except Exception as e:
        print(f"Error during iteration: {e}")

if __name__ == "__main__":
    main()
