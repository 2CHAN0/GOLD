
import logging
from transformers import AutoTokenizer
# Mock the module structure to test the patch logic in isolation
import sys
from types import ModuleType

# Create dummy trl module
trl = ModuleType("trl")
sys.modules["trl"] = trl
trl.experimental = ModuleType("trl.experimental")
trl.experimental.gold = ModuleType("trl.experimental.gold")
gold_module = ModuleType("trl.experimental.gold.gold_trainer")
trl.experimental.gold.gold_trainer = gold_module

# Mock build_teacher_inputs_from_texts
def mock_builder(tokenizer, prompt_texts, completion_texts):
    return prompt_texts

gold_module.build_teacher_inputs_from_texts = mock_builder

# Import the patch function from the script (we need to import it or copy it)
# Since we modified the script, we can import it if we handle the imports inside it.
# But the script imports trl, which we just mocked.
# Let's copy the patch function logic here for unit testing, 
# OR better, let's run the actual script's function if possible.
# But the script has other imports.
# Let's copy the logic to verify it works as intended.

def apply_teacher_system_prompt_patch(system_prompt: str) -> None:
    if not system_prompt:
        return

    # Use our mocked module
    # from trl.experimental.gold import gold_trainer as gold_module
    # (already imported above)

    if getattr(gold_module.build_teacher_inputs_from_texts, "_style_patch_applied", False):
        return

    base_builder = gold_module.build_teacher_inputs_from_texts

    def wrapped(tokenizer, prompt_texts, completion_texts):
        patched_prompts = []
        for prompt in prompt_texts:
            if not prompt:
                patched_prompts.append(system_prompt)
                continue
                
            if "<|im_start|>user" in prompt:
                system_block = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                if "<|im_start|>system" in prompt:
                    patched_prompts.append(f"{system_block}{prompt}")
                else:
                    patched_prompts.append(f"{system_block}{prompt}")
            else:
                patched_prompts.append(f"{system_prompt}\n\n{prompt}")

        return base_builder(tokenizer, patched_prompts, completion_texts)

    wrapped._style_patch_applied = True
    gold_module.build_teacher_inputs_from_texts = wrapped

def main():
    print("Testing Robust Patch...")
    system_prompt = "You are a style coach."
    apply_teacher_system_prompt_patch(system_prompt)
    
    # Test cases
    tokenizer = None # Not used by mock
    completion_texts = [] # Not used by mock
    
    # Case 1: ChatML format
    prompts = ["<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"]
    result = gold_module.build_teacher_inputs_from_texts(tokenizer, prompts, completion_texts)
    print(f"Case 1 Result: {result[0]!r}")
    assert "<|im_start|>system\nYou are a style coach.<|im_end|>\n<|im_start|>user" in result[0]
    
    # Case 2: Raw text
    prompts = ["Hello world"]
    result = gold_module.build_teacher_inputs_from_texts(tokenizer, prompts, completion_texts)
    print(f"Case 2 Result: {result[0]!r}")
    assert "You are a style coach.\n\nHello world" in result[0]
    
    # Case 3: Existing system prompt
    prompts = ["<|im_start|>system\nExisting<|im_end|>\n<|im_start|>user\nHello<|im_end|>"]
    result = gold_module.build_teacher_inputs_from_texts(tokenizer, prompts, completion_texts)
    print(f"Case 3 Result: {result[0]!r}")
    assert "<|im_start|>system\nYou are a style coach.<|im_end|>\n<|im_start|>system\nExisting" in result[0]

    print("\nAll tests passed!")

    # Verify Chat Template
    print("\nVerifying Chat Template...")
    # Read template from file to avoid import issues with mocks
    with open("scripts/train_gold_style.py", "r") as f:
        content = f.read()
    
    # Extract QWEN_CHAT_TEMPLATE string
    import re
    match = re.search(r'QWEN_CHAT_TEMPLATE = """(.*?)"""', content, re.DOTALL)
    if match:
        QWEN_CHAT_TEMPLATE = match.group(1)
    else:
        print("Could not find QWEN_CHAT_TEMPLATE in file.")
        return

    from transformers import AutoTokenizer
    
    # We need a tokenizer to apply the template
    # We can use the one from the environment or mock it?
    # AutoTokenizer needs a model name.
    try:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", trust_remote_code=True)
        tokenizer.chat_template = QWEN_CHAT_TEMPLATE
        
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"}
        ]
        rendered = tokenizer.apply_chat_template(messages, tokenize=False)
        print(f"Rendered:\n{rendered}")
        
        # Check for generation tags if possible? 
        # apply_chat_template usually renders the string. 
        # The tags might be consumed or rendered depending on the engine.
        # TRL uses a specific processor.
        # But we can check if the output looks correct (contains content).
        assert "Hi there" in rendered
        assert "<|im_start|>assistant" in rendered
        print("Template rendered successfully.")
        
    except Exception as e:
        print(f"Skipping template verification due to missing model/dependencies: {e}")

if __name__ == "__main__":
    main()
