from train_gold_style import dynamic_prompt_generator

def main():
    print("Testing dynamic_prompt_generator...")
    gen = dynamic_prompt_generator(seed=42, chosun_prob=0.5, student_system_prompt="")
    
    item = next(gen)
    messages = item["messages"]
    print(f"Generated messages: {messages}")
    
    roles = [msg["role"] for msg in messages]
    assert "user" in roles, "User message missing!"
    assert "assistant" in roles, "Assistant message missing!"
    
    # Check content
    user_msg = next(msg for msg in messages if msg["role"] == "user")
    assert user_msg["content"], "User content is empty!"
    
    print("Generator verification passed!")

if __name__ == "__main__":
    main()
