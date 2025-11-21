from transformers import AutoTokenizer

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print("Chat Template:", tokenizer.chat_template)
    
    messages = [{"role": "system", "content": "SYSTEM_PROMPT"}, {"role": "user", "content": "USER_PROMPT"}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print("\nFormatted Example:\n", formatted)
except Exception as e:
    print(e)
