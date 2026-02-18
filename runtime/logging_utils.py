"""
Logging and cost estimation.

WHY THIS IS THE MOST IMPORTANT FILE:
This is where ROI is MEASURED. Every model call gets logged with:
- Which model ran it
- Was it edge or cloud?
- How long did it take?
- How many tokens were used?
- How much did it cost (or $0 for edge)?

Without this data, you have no ROI story.
With it, you have a dashboard, empirical evidence, and
a compelling narrative for every interview.
"""

import sqlite3
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict

# --- Cost Table ---
# Prices per 1 MILLION tokens (input/output)
# Ollama is always $0 — the whole point of edge compute.
# Cloud prices from Google's published rates.
COST_TABLE = {
    "ollama/mistral": {"input": 0.0, "output": 0.0},
    "ollama/phi3": {"input": 0.0, "output": 0.0},
    "gemini/gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "gemini/gemini-1.5-pro": {"input": 1.25, "output": 5.00},
}


def estimate_cost(model_name: str, prompt_tokens: int,
                  completion_tokens: int) -> float:
    """
    Calculate the USD cost of a model call.

    HOW TOKEN PRICING WORKS:
    Cloud APIs charge per token. A "token" is roughly 4 characters
    or 3/4 of a word. Prices are quoted per 1 million tokens.

    Example: Gemini Flash at $0.075/1M input tokens
    If your prompt is 1,000 tokens:
    Cost = (1000 / 1,000,000) * 0.075 = $0.000075

    Local models (Ollama) cost $0 because they run on YOUR hardware.
    You already paid for the M1 Pro — marginal cost is just electricity.
    """
    if model_name not in COST_TABLE:
        return 0.0
    prices = COST_TABLE[model_name]
    input_cost = (prompt_tokens / 1_000_000) * prices["input"]
    output_cost = (completion_tokens / 1_000_000) * prices["output"]
    return round(input_cost + output_cost, 8)


def estimate_cost_if_cloud(prompt_tokens: int, completion_tokens: int,
                           cloud_model: str = "gemini/gemini-1.5-flash") -> float:
    """
    What WOULD this have cost if run on cloud?

    WHY THIS FUNCTION EXISTS:
    This is the killer metric. For every local (edge) call, we also
    calculate the hypothetical cloud cost. The difference between
    actual cost and hypothetical cloud cost = MONEY SAVED.
    This is your ROI number.
    """
    return estimate_cost(cloud_model, prompt_tokens, completion_tokens)


class CallLogger:
    """
    Logs all model calls to SQLite.

    WHAT IS SQLITE?
    SQLite is a database engine that stores everything in a single file
    (like logs.db). Unlike MySQL or PostgreSQL, it needs NO server —
    no Docker, no configuration, no setup. Python has built-in support.

    WHY NOT JUST A CSV?
    With SQLite, you can write SQL queries to analyze your data:
    "SELECT model_name, AVG(latency_ms) FROM call_logs GROUP BY model_name"
    Try doing that with a CSV — you'd need pandas or manual parsing.

    PYCHARM BONUS:
    PyCharm Pro can open SQLite databases directly! You can browse
    tables, run queries, and see your data visually without writing code.
    (View → Tool Windows → Database, then drag in the .db file)
    """

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = Path(__file__).parent.parent / "data" / "logs.db"

        # Create the data directory if it doesn't exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = str(db_path)
        self._init_db()

    def _init_db(self):
        """
        Create the call_logs table if it doesn't already exist.

        WHAT IS SQL?
        SQL (Structured Query Language) is the language for talking
        to databases. CREATE TABLE defines the schema (columns).
        IF NOT EXISTS means "only create it the first time."
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                         CREATE TABLE IF NOT EXISTS call_logs
                         (
                             id
                             INTEGER
                             PRIMARY
                             KEY
                             AUTOINCREMENT,
                             timestamp
                             TEXT
                             NOT
                             NULL,
                             model_name
                             TEXT
                             NOT
                             NULL,
                             tier
                             TEXT
                             NOT
                             NULL,
                             task_type
                             TEXT,
                             latency_ms
                             REAL,
                             prompt_tokens
                             INTEGER,
                             completion_tokens
                             INTEGER,
                             total_tokens
                             INTEGER,
                             estimated_cost_usd
                             REAL,
                             cloud_alternative_cost_usd
                             REAL,
                             privacy_enforced
                             BOOLEAN
                             DEFAULT
                             0,
                             budget_enforced
                             BOOLEAN
                             DEFAULT
                             0,
                             success
                             BOOLEAN
                             DEFAULT
                             1,
                             error
                             TEXT
                         )
                         """)
            conn.commit()

    def log(self, model_name: str, tier: str, task_type: str,
            latency_ms: float, prompt_tokens: int, completion_tokens: int,
            success: bool = True, error: Optional[str] = None,
            privacy_enforced: bool = False, budget_enforced: bool = False):
        """Log a single model call to the database."""

        cost = estimate_cost(
            model_name, prompt_tokens or 0, completion_tokens or 0
        )
        cloud_alt_cost = estimate_cost_if_cloud(
            prompt_tokens or 0, completion_tokens or 0
        )

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                         INSERT INTO call_logs
                         (timestamp, model_name, tier, task_type, latency_ms,
                          prompt_tokens, completion_tokens, total_tokens,
                          estimated_cost_usd, cloud_alternative_cost_usd,
                          privacy_enforced, budget_enforced, success, error)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                         """, (
                             datetime.utcnow().isoformat(),
                             model_name, tier, task_type, latency_ms,
                             prompt_tokens, completion_tokens,
                             (prompt_tokens or 0) + (completion_tokens or 0),
                             cost, cloud_alt_cost,
                             privacy_enforced, budget_enforced, success, error
                         ))
            conn.commit()

    def get_month_to_date_cost(self) -> float:
        """
        Total cloud spend this month.
        Used by budget enforcement in the router.
        """
        month_start = (datetime.utcnow()
                       .replace(day=1, hour=0, minute=0, second=0)
                       .isoformat())
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute("""
                                  SELECT COALESCE(SUM(estimated_cost_usd), 0)
                                  FROM call_logs
                                  WHERE timestamp >= ? AND tier = 'cloud'
                                  """, (month_start,)).fetchone()
        return result[0]

    def get_summary_stats(self) -> Dict:
        """
        Aggregate stats for the dashboard and ROI report.

        This is the function that powers your entire ROI narrative.
        It answers: how many calls, how much spent, how much saved?
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            overall = conn.execute("""
                                   SELECT COUNT(*)                as total_calls,
                                          SUM(CASE WHEN tier = 'edge' THEN 1 ELSE 0 END)
                                                                  as edge_calls,
                                          SUM(CASE WHEN tier = 'cloud' THEN 1 ELSE 0 END)
                                                                  as cloud_calls,
                                          SUM(estimated_cost_usd) as total_cost,
                                          SUM(cloud_alternative_cost_usd)
                                                                  as total_cloud_alternative_cost,
                                          AVG(latency_ms)         as avg_latency,
                                          SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END)
                                              * 100.0 / COUNT(*)  as success_rate
                                   FROM call_logs
                                   """).fetchone()

            per_model = conn.execute("""
                                     SELECT model_name,
                                            tier,
                                            COUNT(*)                as calls,
                                            AVG(latency_ms)         as avg_latency_ms,
                                            SUM(estimated_cost_usd) as total_cost,
                                            SUM(total_tokens)       as total_tokens
                                     FROM call_logs
                                     GROUP BY model_name
                                     """).fetchall()

            overall_dict = dict(overall)

            return {
                "overall": overall_dict,
                "per_model": [dict(row) for row in per_model],
                "cost_avoided": (
                        (overall_dict["total_cloud_alternative_cost"] or 0) -
                        (overall_dict["total_cost"] or 0)
                ),
                "edge_percentage": (
                        (overall_dict["edge_calls"] or 0) /
                        max(overall_dict["total_calls"], 1) * 100
                ),
            }
