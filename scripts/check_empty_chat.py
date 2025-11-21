from transformers import AutoTokenizer

try:
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", trust_remote_code=True)
    print("Attempting apply_chat_template([])")
    tokenizer.apply_chat_template([])
    print("Success (unexpected)")
except IndexError as e:
    print(f"Caught expected IndexError: {e}")
except Exception as e:
    print(f"Caught unexpected exception: {type(e).__name__}: {e}")
