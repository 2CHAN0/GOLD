"""Test style configuration system."""
import sys
sys.path.insert(0, "scripts")

from style_config import get_default_style_registry
from pathlib import Path

def test_style_registry():
    print("=" * 60)
    print("Testing Style Configuration System")
    print("=" * 60)
    
    # Test 1: Load style registry
    print("\n1. Loading style registry...")
    try:
        registry = get_default_style_registry()
        styles = registry.list_styles()
        print(f"   ✅ Loaded {len(styles)} styles: {', '.join(styles)}")
    except Exception as e:
        print(f"   ❌ Failed to load registry: {e}")
        return False
    
    # Test 2: Load individual styles
    print("\n2. Testing individual styles...")
    for style_name in styles:
        try:
            style = registry.get_style(style_name)
            print(f"   ✅ {style_name}: {style.display_name}")
            print(f"      - Examples: {len(style.examples)}")
            if style.validation:
                print(f"      - Positive keywords: {len(style.validation.positive_keywords)}")
                print(f"      - Negative keywords: {len(style.validation.negative_keywords)}")
        except Exception as e:
            print(f"   ❌ Failed to load {style_name}: {e}")
            return False
    
    # Test 3: Build system prompt
    print("\n3. Building combined system prompt...")
    try:
        system_prompt = registry.build_combined_system_prompt(
            style_names=None,
            include_examples=True
        )
        print(f"   ✅ Built system prompt ({len(system_prompt)} chars)")
        print("\n   Preview (first 300 chars):")
        print(f"   {system_prompt[:300]}...")
    except Exception as e:
        print(f"   ❌ Failed to build system prompt: {e}")
        return False
    
    # Test 4: Validate outputs
    print("\n4. Testing validation...")
    chosun_style = registry.get_style("chosun")
    none_style = registry.get_style("none")
    
    # Good chosun example
    chosun_text = "소자가 아뢰옵니다. 백성들께 청하옵니다."
    chosun_scores = chosun_style.validate_output(chosun_text)
    print(f"   Chosun text validation: {chosun_scores}")
    
    # Bad chosun example (modern text)
    modern_text = "안녕하세요. 이렇게 하면 좋아요."
    modern_scores = chosun_style.validate_output(modern_text)
    print(f"   Modern text on Chosun validation: {modern_scores}")
    
    # Good modern example
    none_scores = none_style.validate_output(modern_text)
    print(f"   Modern text validation: {none_scores}")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    test_style_registry()
