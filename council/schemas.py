"""
Schemas for the Council of Local LLMs.

THREE PHASES OF DELIBERATION, THREE SCHEMAS:

1. ROUTING VOTE — Should this prompt go to cloud or stay local?
2. MODEL SELECTION VOTE — If local, which model should answer?
3. ANSWER REVIEW — Is this answer good enough, or does it need work?

Each schema defines the exact JSON structure that council members
must output. This is CRITICAL because we're parsing 5 different
models' outputs programmatically — if the JSON is wrong, the
deliberation breaks.
"""

# --- Phase 1: Routing Vote ---
# Each model votes on whether the prompt needs cloud or can be local.
ROUTING_VOTE_SCHEMA = {
    "type": "object",
    "required": ["vote", "reasoning", "confidence"],
    "properties": {
        "vote": {
            "type": "string",
            "enum": ["local", "cloud"],
            "description": "Whether to handle locally or send to cloud"
        },
        "reasoning": {
            "type": "string",
            "description": "Why this vote was cast — 2-3 sentences max"
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "How confident the model is in this vote"
        },
        "suggested_cloud_tier": {
            "type": "string",
            "enum": ["budget", "standard", "premium", "luxury", "none"],
            "description": "If voting cloud, which tier. 'none' if local."
        }
    }
}


# --- Phase 2: Model Selection Vote ---
# Each model votes on which local model should generate the answer.
MODEL_SELECTION_SCHEMA = {
    "type": "object",
    "required": ["selected_model", "reasoning", "confidence"],
    "properties": {
        "selected_model": {
            "type": "string",
            "description": "The model ID that should answer this prompt"
        },
        "reasoning": {
            "type": "string",
            "description": "Why this model is the best choice"
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        }
    }
}


# --- Phase 3: Answer Review ---
# Each model reviews the generated answer and votes approve/reject.
ANSWER_REVIEW_SCHEMA = {
    "type": "object",
    "required": ["approved", "quality_score", "feedback"],
    "properties": {
        "approved": {
            "type": "boolean",
            "description": "True if the answer meets quality standards"
        },
        "quality_score": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Overall quality rating"
        },
        "feedback": {
            "type": "string",
            "description": "Specific, actionable feedback for improvement. "
                          "Empty string if approved."
        },
        "issues": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of specific issues found"
        }
    }
}


# --- Final Deliberation Output ---
# The complete record of a council deliberation.
DELIBERATION_RESULT_SCHEMA = {
    "type": "object",
    "required": ["final_answer", "routing_decision", "selected_model",
                  "iterations", "consensus_reached"],
    "properties": {
        "final_answer": {"type": "string"},
        "routing_decision": {
            "type": "string",
            "enum": ["local", "cloud"]
        },
        "selected_model": {"type": "string"},
        "iterations": {"type": "integer", "minimum": 1},
        "consensus_reached": {"type": "boolean"},
        "routing_votes": {"type": "object"},
        "review_summary": {"type": "object"},
        "total_deliberation_ms": {"type": "number"},
    }
}
