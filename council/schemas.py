"""
Strict JSON schema for council outputs.

WHY SCHEMAS?
Without enforcement, AI outputs are unpredictable strings.
With schemas, you get STRUCTURED, PARSEABLE, TESTABLE data.
"""

COUNCIL_OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["summary", "assumptions", "risks",
                  "disagreements", "confidence"],
    "properties": {
        "summary": {"type": "string"},
        "assumptions": {
            "type": "array", "items": {"type": "string"}
        },
        "risks": {
            "type": "array", "items": {"type": "string"}
        },
        "disagreements": {
            "type": "array", "items": {"type": "string"}
        },
        "confidence": {
            "type": "number", "minimum": 0.0, "maximum": 1.0
        },
    }
}
