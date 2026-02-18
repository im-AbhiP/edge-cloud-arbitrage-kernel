"""
Prompt registry — loads templates and their expected token counts.

WHY THIS FILE EXISTS:
1. Centralizes all prompts so they're version-controlled and consistent
2. Each prompt has expected_tokens, which the router uses to PREDICT
   cost BEFORE making an API call. This is predictive cost governance.
3. YAML format is human-readable and easy to edit without touching
   Python code — a product manager could update prompts without
   knowing Python.
"""

import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class PromptTemplate:
    """One prompt template with its metadata."""
    name: str
    template: str
    expected_tokens: int
    task_type: str


class PromptRegistry:
    """
    Loads and manages prompt templates from a YAML file.

    WHAT IS A REGISTRY PATTERN?
    A registry is a central place to look up objects by name.
    Instead of hardcoding prompts throughout your code,
    you register them once here and look them up by key.
    """

    def __init__(self, yaml_path: Optional[str] = None):
        if yaml_path is None:
            # Path(__file__).parent = the folder containing THIS file
            # So this looks for prompts.yaml in the same folder
            yaml_path = Path(__file__).parent / "prompts.yaml"

        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        self._prompts: Dict[str, PromptTemplate] = {}
        for key, val in data.get("prompts", {}).items():
            self._prompts[key] = PromptTemplate(
                name=val["name"],
                template=val["template"],
                expected_tokens=val["expected_tokens"],
                task_type=val["task_type"],
            )

    def get(self, name: str) -> PromptTemplate:
        """Look up a prompt by its key name."""
        if name not in self._prompts:
            raise KeyError(
                f"Prompt '{name}' not found. "
                f"Available: {list(self._prompts.keys())}"
            )
        return self._prompts[name]

    def render(self, name: str, **kwargs) -> str:
        """
        Get a prompt and fill in its variables.

        Example:
            registry.render("quick_qa", question="What is TCP?")
            → "Answer concisely: What is TCP?"
        """
        template = self.get(name)
        return template.template.format(**kwargs)

    def list_prompts(self) -> list:
        """Return all available prompt names."""
        return list(self._prompts.keys())
