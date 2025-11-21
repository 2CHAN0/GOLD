from transformers import AutoTokenizer

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
with open("qwen_template.jinja", "w") as f:
    f.write(tokenizer.chat_template)
