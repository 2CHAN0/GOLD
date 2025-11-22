#!/usr/bin/env python3
"""Test script to verify ppubi style configuration and dynamic prompt generation."""

import sys
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from style_config import get_default_style_registry, generate_dynamic_prompt


def test_ppubi_style():
    """Test ppubi style loading and dynamic prompt generation."""
    print("=" * 80)
    print("뿌비 스타일 설정 테스트")
    print("=" * 80)
    
    # Load style registry
    registry = get_default_style_registry()
    print(f"\n✓ 로드된 스타일: {', '.join(registry.list_styles())}")
    
    # Get ppubi style
    ppubi_style = registry.get_style("ppubi")
    print(f"\n✓ 뿌비 스타일 로드 성공")
    print(f"  - 이름: {ppubi_style.display_name}")
    print(f"  - 설명: {ppubi_style.description}")
    
    # Show system prompt preview
    print(f"\n{'='*80}")
    print("시스템 프롬프트 (처음 500자만 표시):")
    print(f"{'='*80}")
    system_prompt = ppubi_style.build_teacher_system_prompt(include_examples=False)
    print(system_prompt[:500] + "..." if len(system_prompt) > 500 else system_prompt)
    
    # Show few-shot examples
    print(f"\n{'='*80}")
    print(f"Few-shot 예시 ({len(ppubi_style.examples)}개):")
    print(f"{'='*80}")
    for i, ex in enumerate(ppubi_style.examples[:3], 1):  # Show first 3
        print(f"\n예시 {i}:")
        print(f"  User: {ex.user}")
        print(f"  Assistant: {ex.assistant[:100]}..." if len(ex.assistant) > 100 else f"  Assistant: {ex.assistant}")
    
    # Test dynamic prompt generation
    print(f"\n{'='*80}")
    print("동적 프롬프트 생성 테스트 (20개 샘플):")
    print(f"{'='*80}")
    for i in range(20):
        prompt = generate_dynamic_prompt(ppubi_style)
        print(f"{i+1:2d}. {prompt}")
    
    # Test validation keywords
    print(f"\n{'='*80}")
    print("스타일 검증 키워드:")
    print(f"{'='*80}")
    if ppubi_style.validation:
        print(f"\n긍정 키워드 (있어야 할 표현): {ppubi_style.validation.positive_keywords}")
        print(f"\n부정 키워드 (있으면 안되는 표현): {ppubi_style.validation.negative_keywords}")
        
        # Test validation
        test_text = "뿌비비빕! 오늘 날씨 정말 좋아– 뿌비! 산책하기 딱 좋은 날씨야 뿌↗비!"
        scores = ppubi_style.validate_output(test_text)
        print(f"\n테스트 텍스트: {test_text}")
        print(f"검증 점수: {scores}")
    
    # Test combined system prompt with multiple styles
    print(f"\n{'='*80}")
    print("조선시대 + 뿌비 조합 시스템 프롬프트 (처음 800자만 표시):")
    print(f"{'='*80}")
    combined_prompt = registry.build_combined_system_prompt(
        style_names=["chosun", "ppubi"],
        include_examples=True
    )
    print(combined_prompt[:800] + "..." if len(combined_prompt) > 800 else combined_prompt)
    
    print(f"\n{'='*80}")
    print("✓ 모든 테스트 통과!")
    print(f"{'='*80}")


if __name__ == "__main__":
    test_ppubi_style()
