"""Style configuration loader and utilities for GOLD training.

This module provides utilities to load style configurations from YAML files,
which include system prompts, few-shot examples, and validation keywords.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import yaml


@dataclass
class StyleExample:
    """Single few-shot example for a style."""
    user: str
    assistant: str


@dataclass
class StyleValidation:
    """Validation keywords for a style."""
    positive_keywords: List[str]
    negative_keywords: List[str]


@dataclass
class StyleConfig:
    """Complete configuration for a single style."""
    name: str
    display_name: str
    description: str
    system_prompt: str
    examples: List[StyleExample]
    validation: Optional[StyleValidation] = None
    dynamic_prompt_templates: Optional[Dict] = None
    
    @classmethod
    def from_yaml(cls, path: Path) -> "StyleConfig":
        """Load style config from YAML file.
        
        Args:
            path: Path to YAML file
            
        Returns:
            StyleConfig instance
        """
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        # Parse examples
        examples = []
        for ex in data.get("examples", []):
            examples.append(StyleExample(
                user=ex["user"],
                assistant=ex["assistant"]
            ))
        
        # Parse validation
        validation = None
        if "validation" in data:
            val_data = data["validation"]
            validation = StyleValidation(
                positive_keywords=val_data.get("positive_keywords", []),
                negative_keywords=val_data.get("negative_keywords", [])
            )
        
        return cls(
            name=data["name"],
            display_name=data.get("display_name", data["name"]),
            description=data.get("description", ""),
            system_prompt=data["system_prompt"],
            examples=examples,
            validation=validation,
            dynamic_prompt_templates=data.get("dynamic_prompt_templates")
        )
    
    def build_teacher_system_prompt(self, include_examples: bool = True) -> str:
        """Build complete system prompt for teacher model.
        
        Args:
            include_examples: Whether to include few-shot examples
            
        Returns:
            Complete system prompt string
        """
        parts = [self.system_prompt.strip()]
        
        if include_examples and self.examples:
            parts.append("\n\n**예시**:")
            for i, ex in enumerate(self.examples, 1):
                parts.append(f"\n{i}. 입력: {ex.user}")
                parts.append(f"   출력: {ex.assistant}")
        
        return "\n".join(parts)
    
    def validate_output(self, text: str) -> Dict[str, float]:
        """Validate if output follows the style.
        
        Args:
            text: Generated text to validate
            
        Returns:
            Dict with scores: positive_score, negative_score, overall_score
        """
        if not self.validation:
            return {"positive_score": 1.0, "negative_score": 0.0, "overall_score": 1.0}
        
        text_lower = text.lower()
        
        # Count positive keywords
        positive_count = sum(
            1 for kw in self.validation.positive_keywords
            if kw.lower() in text_lower
        )
        positive_score = (
            positive_count / len(self.validation.positive_keywords)
            if self.validation.positive_keywords
            else 1.0
        )
        
        # Count negative keywords (should be 0)
        negative_count = sum(
            1 for kw in self.validation.negative_keywords
            if kw.lower() in text_lower
        )
        negative_score = (
            negative_count / len(self.validation.negative_keywords)
            if self.validation.negative_keywords
            else 0.0
        )
        
        # Overall score (positive - negative)
        overall_score = max(0.0, positive_score - negative_score)
        
        return {
            "positive_score": positive_score,
            "negative_score": negative_score,
            "overall_score": overall_score
        }


class StyleRegistry:
    """Registry for managing multiple style configurations."""
    
    def __init__(self, styles_dir: Path):
        """Initialize style registry.
        
        Args:
            styles_dir: Directory containing style YAML files
        """
        self.styles_dir = Path(styles_dir)
        self.styles: Dict[str, StyleConfig] = {}
        self._load_all_styles()
    
    def _load_all_styles(self) -> None:
        """Load all style configurations from directory."""
        if not self.styles_dir.exists():
            raise FileNotFoundError(f"Styles directory not found: {self.styles_dir}")
        
        for yaml_file in self.styles_dir.glob("*.yaml"):
            try:
                config = StyleConfig.from_yaml(yaml_file)
                self.styles[config.name] = config
            except Exception as e:
                print(f"Warning: Failed to load {yaml_file}: {e}")
    
    def get_style(self, name: str) -> StyleConfig:
        """Get style config by name.
        
        Args:
            name: Style name
            
        Returns:
            StyleConfig instance
            
        Raises:
            KeyError: If style not found
        """
        if name not in self.styles:
            available = ", ".join(self.styles.keys())
            raise KeyError(
                f"Style '{name}' not found. Available styles: {available}"
            )
        return self.styles[name]
    
    def list_styles(self) -> List[str]:
        """List all available style names."""
        return list(self.styles.keys())
    
    def build_combined_system_prompt(
        self,
        style_names: Optional[List[str]] = None,
        include_examples: bool = True
    ) -> str:
        """Build combined system prompt for multiple styles.
        
        Args:
            style_names: List of style names to include (None = all)
            include_examples: Whether to include examples
            
        Returns:
            Combined system prompt string
        """
        if style_names is None:
            style_names = self.list_styles()
        
        if len(style_names) == 1:
            return self.get_style(style_names[0]).build_teacher_system_prompt(
                include_examples=include_examples
            )
        
        # Multiple styles: combine them
        parts = [
            "당신은 다양한 한국어 스타일을 구사하는 전문가입니다.",
            "\n프롬프트의 스타일 태그에 따라 적절한 말투로 응답해야 합니다.\n"
        ]
        
        for i, style_name in enumerate(style_names, 1):
            style = self.get_style(style_name)
            parts.append(f"\n## {i}. {style.display_name} (<style:{style.name}>)")
            parts.append(f"{style.description}\n")
            parts.append(style.system_prompt)
            
            if include_examples and style.examples:
                parts.append("\n**예시**:")
                for ex in style.examples[:2]:  # Limit to 2 examples per style
                    parts.append(f"- 입력: {ex.user}")
                    parts.append(f"  출력: {ex.assistant}")
        
        parts.append(
            "\n\n**중요**: 학생 모델이 당신의 응답 패턴을 학습합니다. "
            "각 스타일에 대해 일관성을 유지해야 합니다."
        )
        
        return "\n".join(parts)


def generate_dynamic_prompt(
    style_config: StyleConfig,
    style_tag: Optional[str] = None
) -> str:
    """Generate a random prompt using dynamic templates.
    
    Args:
        style_config: Style configuration
        style_tag: Optional style tag to prepend (e.g., "<style:chosun>")
        
    Returns:
        Generated prompt string
    """
    if not style_config.dynamic_prompt_templates:
        raise ValueError(f"Style '{style_config.name}' has no dynamic templates")
    
    templates = style_config.dynamic_prompt_templates
    
    # Build prompt based on available template keys
    if style_config.name == "chosun":
        recipient = random.choice(templates.get("recipients", [""]))
        theme = random.choice(templates.get("themes", [""]))
        task = random.choice(templates.get("tasks", [""]))
        prompt = f"{recipient} {theme}에 대해 {task}"
    elif style_config.name == "none":
        topic = random.choice(templates.get("topics", [""]))
        task = random.choice(templates.get("tasks", [""]))
        prompt = f"{topic} {task}"
    else:
        # Generic template
        prompt = "질문을 작성해 주세요"
    
    # Add style tag if provided
    if style_tag:
        prompt = f"{style_tag} {prompt}"
    elif style_config.name:
        prompt = f"<style:{style_config.name}> {prompt}"
    
    return prompt.strip()


# Convenience function for loading default registry
_default_registry: Optional[StyleRegistry] = None


def get_default_style_registry(styles_dir: Optional[Path] = None) -> StyleRegistry:
    """Get or create default style registry.
    
    Args:
        styles_dir: Optional custom styles directory
        
    Returns:
        StyleRegistry instance
    """
    global _default_registry
    
    if _default_registry is None or styles_dir is not None:
        if styles_dir is None:
            # Default to prompts/styles in project root
            styles_dir = Path(__file__).parent.parent / "prompts" / "styles"
        _default_registry = StyleRegistry(styles_dir)
    
    return _default_registry
