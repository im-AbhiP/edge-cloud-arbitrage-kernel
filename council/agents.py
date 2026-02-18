"""
Multi-agent research council.

WHY MULTIPLE AGENTS?
A single LLM call gives one perspective. Three agents with different
system prompts give you STRUCTURED DISAGREEMENT — surfacing
assumptions, risks, and blind spots that a single call would miss.
"""

import json
from typing import Optional
from runtime.tasks import TaskMetadata, TaskType, DataSensitivity
from runtime.router import ModelRouter

EXPLAINER_PROMPT = """You are the Explainer agent. Provide a clear, 
thorough analysis. Focus on key concepts, context, and nuances."""

SKEPTIC_PROMPT = """You are the Skeptic agent. Critically examine 
the topic. Focus on hidden assumptions, flaws, alternative views, 
and risks."""

SYNTHESIZER_PROMPT = """You are the Synthesizer. Merge the Explainer's 
and Skeptic's analyses into final structured output.

Output ONLY valid JSON with this exact schema:
{
    "summary": "Balanced synthesis",
    "assumptions": ["assumption1", "assumption2"],
    "risks": ["risk1", "risk2"],
    "disagreements": ["disagreement1"],
    "confidence": 0.0 to 1.0
}

Output ONLY the JSON. No markdown, no explanation."""


class ResearchCouncil:
    """Orchestrates three agents to produce structured output."""

    def __init__(self, router: Optional[ModelRouter] = None):
        self.router = router or ModelRouter()

    def run(self, question: str,
            data_sensitivity: DataSensitivity = DataSensitivity.LOW
            ) -> dict:
        """
        Full council pipeline:
        1. Explainer analyzes
        2. Skeptic critiques
        3. Synthesizer merges into JSON
        """

        print("[Council] Phase 1: Explainer...")
        explainer_meta = TaskMetadata(
            task_type=TaskType.DEEP_RESEARCH,
            complexity=0.7, importance=0.6,
            data_sensitivity=data_sensitivity,
        )
        explainer_result = self.router.run(
            prompt=f"Analyze thoroughly: {question}",
            meta=explainer_meta,
            system_prompt=EXPLAINER_PROMPT,
            max_tokens=1500,
        )

        print("[Council] Phase 2: Skeptic...")
        skeptic_meta = TaskMetadata(
            task_type=TaskType.DEEP_RESEARCH,
            complexity=0.7, importance=0.6,
            data_sensitivity=data_sensitivity,
        )
        skeptic_result = self.router.run(
            prompt=(f"Topic: {question}\n\n"
                    f"Explainer's analysis:\n{explainer_result.text}\n\n"
                    f"Provide your critical analysis."),
            meta=skeptic_meta,
            system_prompt=SKEPTIC_PROMPT,
            max_tokens=1500,
        )

        print("[Council] Phase 3: Synthesizer...")
        synth_meta = TaskMetadata(
            task_type=TaskType.DEEP_RESEARCH,
            complexity=0.8, importance=0.9,
            data_sensitivity=data_sensitivity,
        )
        synth_result = self.router.run(
            prompt=(f"Topic: {question}\n\n"
                    f"Explainer:\n{explainer_result.text}\n\n"
                    f"Skeptic:\n{skeptic_result.text}\n\n"
                    f"Synthesize into JSON."),
            meta=synth_meta,
            system_prompt=SYNTHESIZER_PROMPT,
            max_tokens=1000,
        )

        # Parse JSON
        try:
            text = synth_result.text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            output = json.loads(text)
        except json.JSONDecodeError:
            output = {
                "summary": synth_result.text,
                "assumptions": ["Could not parse structured output"],
                "risks": ["Synthesizer did not produce valid JSON"],
                "disagreements": [],
                "confidence": 0.0,
                "_parse_error": True,
            }

        return {
            "council_output": output,
            "explainer_response": explainer_result.text,
            "skeptic_response": skeptic_result.text,
            "models_used": {
                "explainer": explainer_result.model_name,
                "skeptic": skeptic_result.model_name,
                "synthesizer": synth_result.model_name,
            }
        }
