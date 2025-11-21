"""Test chat template with assistant token masking."""
from transformers import AutoTokenizer
import sys
sys.path.append('scripts')
from train_gold_style import QWEN_CHAT_TEMPLATE

def test_assistant_mask():
    print("=" * 60)
    print("Testing Assistant Token Masking")
    print("=" * 60)
    
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        trust_remote_code=True
    )
    
    # Set our custom template
    tokenizer.chat_template = QWEN_CHAT_TEMPLATE
    
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"}
    ]
    
    print("\n📝 Test Messages:")
    for msg in messages:
        print(f"  {msg['role']}: {msg['content']}")
    
    # Test with return_assistant_tokens_mask
    print("\n🔍 Testing return_assistant_tokens_mask=True...")
    try:
        result = tokenizer.apply_chat_template(
            messages,
            return_dict=True,
            return_assistant_tokens_mask=True,
            tokenize=True
        )
        
        print(f"\n✅ Success! Keys returned: {list(result.keys())}")
        
        if "input_ids" in result:
            print(f"   - input_ids shape: {result['input_ids'].shape if hasattr(result['input_ids'], 'shape') else len(result['input_ids'])}")
        
        if "assistant_masks" in result:
            print(f"   - assistant_masks found!")
            print(f"   - assistant_masks shape: {result['assistant_masks'].shape if hasattr(result['assistant_masks'], 'shape') else len(result['assistant_masks'])}")
            print(f"   - assistant_masks: {result['assistant_masks']}")
        elif "labels" in result:
            print(f"   - labels found (TRL uses this for masking)")
            print(f"   - labels shape: {result['labels'].shape if hasattr(result['labels'], 'shape') else len(result['labels'])}")
        else:
            print("   ⚠️  WARNING: No assistant_masks or labels returned!")
            print("   This means assistant-only-loss may not work properly.")
            
        return result
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_template_parsing():
    print("\n" + "=" * 60)
    print("Testing Template Parsing")
    print("=" * 60)
    
    try:
        import jinja2
        env = jinja2.Environment()
        env.parse(QWEN_CHAT_TEMPLATE)
        print("✅ Template parses successfully with standard Jinja2!")
    except jinja2.TemplateSyntaxError as e:
        print(f"❌ Template syntax error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
    
    return True

def test_formatted_output():
    print("\n" + "=" * 60)
    print("Testing Formatted Output")
    print("=" * 60)
    
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        trust_remote_code=True
    )
    tokenizer.chat_template = QWEN_CHAT_TEMPLATE
    
    messages = [
        {"role": "user", "content": "<style:none> 봄맞이 집안 정리 요령 몇 가지를 쉽게 알려 줘."},
        {"role": "assistant", "content": "1. 청소의 일정표를 짜서 매일 실시하는 것이 좋다."}
    ]
    
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    
    print("\n📝 Formatted conversation:")
    print(formatted)
    print("\n✅ Formatting complete!")

if __name__ == "__main__":
    # Run all tests
    success = True
    
    # Test 1: Template parsing
    if not test_template_parsing():
        success = False
    
    # Test 2: Assistant mask
    result = test_assistant_mask()
    if result is None:
        success = False
    
    # Test 3: Formatted output
    test_formatted_output()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed. Please review the output above.")
    print("=" * 60)
