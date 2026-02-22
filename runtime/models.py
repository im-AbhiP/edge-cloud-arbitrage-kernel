"""
Model client abstraction layer.

WHY THIS FILE EXISTS:
We need to talk to multiple AI backends (Ollama local, Gemini cloud)
through a UNIFORM interface. This means:
- The router doesn't care WHICH model it's calling
- Logging is consistent regardless of backend
- Adding a new model = adding one new class

This is the STRATEGY PATTERN from software engineering:
define a family of algorithms (model clients), encapsulate each one,
and make them interchangeable.
"""

import time
import os
import json
from dataclasses import dataclass
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
import httpx
from dotenv import load_dotenv

# Load environment variables from .env file
# This reads your GEMINI_API_KEY, PRIVACY_MODE, etc.
load_dotenv()


@dataclass
class ModelResult:
    """
    Uniform result from ANY model call.

    TIMING FIELDS EXPLAINED:
    - latency_ms: Total wall-clock time measured from Python
      (includes network overhead, Ollama processing, etc.)
    - prefill_ms: Time the model spent processing the INPUT prompt.
      This is "reading comprehension" speed. Measured by Ollama internally.
    - decode_ms: Time the model spent GENERATING output tokens.
      This is "writing" speed. Measured by Ollama internally.
    - load_ms: Time spent loading the model into RAM.
      On first call or after model swap, this can be several seconds.

    WHY SEPARATE PREFILL AND DECODE?
    In LLM inference, there are two distinct phases:
    1. PREFILL (also called "prompt processing" or "encoding"):
       The model reads and processes all input tokens at once.
       This is heavily parallelizable and benefits from GPU bandwidth.
    2. DECODE (also called "generation" or "autoregressive"):
       The model generates output tokens one at a time.
       Each token depends on all previous tokens, so it's sequential.

    These have very different performance characteristics:
    - Prefill is bandwidth-bound (how fast can you read?)
    - Decode is latency-bound (how fast can you write one token?)

    At AMD, this distinction matters because their MI300X has
    different advantages for prefill vs decode workloads.
    Showing you understand this in your dashboard is a strong signal.
    """
    text: str
    model_name: str
    tier: str
    latency_ms: float
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    raw_response: Optional[Dict[str, Any]] = None
    success: bool = True
    error: Optional[str] = None
    # NEW: Detailed inference timing from Ollama
    prefill_ms: Optional[float] = None
    decode_ms: Optional[float] = None
    load_ms: Optional[float] = None



@dataclass
class ModelProfile:
    """
    Describes a model's strengths and characteristics.
    The council uses this to decide which model should answer.

    WHY THIS EXISTS:
    When 5 models vote on who should answer, they need to know
    each other's strengths. This is like a team where each member
    knows what the others are good at.
    """
    name: str
    tier: str
    strengths: str
    weaknesses: str
    ram_gb: float
    speed: str  # "fast", "medium", "slow"


# Model profiles for council deliberation
MODEL_PROFILES = {
    "ollama/gpt-oss-20b": ModelProfile(
        name="gpt-oss-20b",
        tier="edge",
        strengths="Strongest reasoning, chain-of-thought, tool calling, "
                  "complex multi-step analysis. OpenAI's MoE architecture.",
        weaknesses="Very large (13GB), slow to load, monopolizes RAM. "
                   "Cannot run alongside other models on 16GB.",
        ram_gb=14.0,
        speed="slow",
    ),
    "ollama/llama3.1-8b": ModelProfile(
        name="llama3.1-8b",
        tier="edge",
        strengths="Strong instruction following, balanced quality and speed, "
                  "good at structured output, reliable all-rounder.",
        weaknesses="Not specialized — decent at everything, best at nothing.",
        ram_gb=6.0,
        speed="medium",
    ),
    "ollama/deepseek-r1-8b": ModelProfile(
        name="deepseek-r1-8b",
        tier="edge",
        strengths="Excellent step-by-step reasoning, analytical thinking, "
                  "good at math and logic problems.",
        weaknesses="Can be verbose in reasoning chains, slower due to "
                   "chain-of-thought process.",
        ram_gb=6.0,
        speed="medium",
    ),
    "ollama/qwen3-8b": ModelProfile(
        name="qwen3-8b",
        tier="edge",
        strengths="Strong multilingual capabilities, good general knowledge, "
                  "solid instruction following, fast responses.",
        weaknesses="Less specialized than domain-specific models.",
        ram_gb=6.0,
        speed="fast",
    ),
    "ollama/deepseek-coder-6.7b": ModelProfile(
        name="deepseek-coder-6.7b",
        tier="edge",
        strengths="Specialized for code generation, review, debugging. "
                  "Smallest model — fastest to load and respond.",
        weaknesses="Weaker on non-code tasks. Smaller parameter count "
                   "means less general knowledge.",
        ram_gb=4.5,
        speed="fast",
    ),
}


class BaseModelClient(ABC):
    """
    Abstract base class that all model clients must implement.

    WHAT IS AN ABSTRACT BASE CLASS (ABC)?
    It's a "contract" or "blueprint." It says: "Any class that inherits
    from me MUST implement these methods." If you forget to implement one,
    Python raises an error at instantiation time, not at runtime.

    Think of it like an interface in Java or a protocol in Swift.
    It guarantees the router can call .call() on ANY model client
    without knowing which specific client it's talking to.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the name of this model, e.g., 'ollama/mistral'."""
        pass

    @property
    @abstractmethod
    def tier(self) -> str:
        """Return 'edge' or 'cloud'."""
        pass

    @abstractmethod
    def call(self, prompt: str, system_prompt: Optional[str] = None,
             max_tokens: int = 1024, temperature: float = 0.7) -> ModelResult:
        """
        Send a prompt to the model and get a response.

        Parameters:
        - prompt: The user's question/task
        - system_prompt: Instructions for how the model should behave
        - max_tokens: Maximum length of the response (in tokens)
        - temperature: Randomness (0.0=deterministic, 1.0=creative)
        """
        pass


class OllamaClient(BaseModelClient):
    """
    Client for local Ollama models.

    HOW OLLAMA WORKS:
    Ollama runs a REST API server on localhost:11434. We send it
    HTTP POST requests with our prompt, and it sends back a response.

    Cost is ALWAYS $0 — this is the entire point of edge compute.
    The tradeoff is that local models are smaller and less capable.
    """

    def __init__(self, model: str = "mistral",
                 base_url: str = "http://localhost:11434"):
        self._model = model
        self.base_url = base_url

    @property
    def model_name(self) -> str:
        return f"ollama/{self._model}"

    @property
    def tier(self) -> str:
        return "edge"

    def call(self, prompt: str, system_prompt: Optional[str] = None,
             max_tokens: int = 1024, temperature: float = 0.7) -> ModelResult:

        # Record the start time so we can measure latency
        start_time = time.time()

        try:
            # Build the request payload (what we send to Ollama)
            payload = {
                "model": self._model,  # Which model to use
                "prompt": prompt,  # The user's text
                "stream": False,  # Get complete response at once
                "options": {
                    "num_predict": max_tokens,  # Max response length
                    "temperature": temperature,  # Randomness level
                }
            }

            # If there's a system prompt, add it
            # System prompts set the model's "personality" or "role"
            if system_prompt:
                payload["system"] = system_prompt

            # Send the HTTP request to Ollama
            # httpx.Client is like a web browser for your code
            # timeout=120 means wait up to 2 minutes for a response
            # (local models on M1 can be slow for long responses)
            with httpx.Client(timeout=120.0) as client:
                response = client.post(
                    f"{self.base_url}/api/generate",
                    json=payload
                )
                # Raise an exception if HTTP status is 4xx or 5xx
                response.raise_for_status()
                data = response.json()

            # Calculate how long the call took
            latency_ms = (time.time() - start_time) * 1000

            # Extract Ollama's internal timing (in nanoseconds)
            # These are much more accurate than our Python-side timing
            # because they measure the actual model computation,
            # not including HTTP overhead.
            prefill_ns = data.get("prompt_eval_duration", 0)
            decode_ns = data.get("eval_duration", 0)
            load_ns = data.get("load_duration", 0)

            # Convert nanoseconds to milliseconds
            # 1 millisecond = 1,000,000 nanoseconds
            prefill_ms = prefill_ns / 1_000_000 if prefill_ns else None
            decode_ms = decode_ns / 1_000_000 if decode_ns else None
            load_ms = load_ns / 1_000_000 if load_ns else None

            return ModelResult(
                text=data.get("response", ""),
                model_name=self.model_name,
                tier=self.tier,
                latency_ms=latency_ms,
                prompt_tokens=data.get("prompt_eval_count"),
                completion_tokens=data.get("eval_count"),
                total_tokens=(
                        data.get("prompt_eval_count", 0)
                        + data.get("eval_count", 0)
                ),
                raw_response=data,
                success=True,
                prefill_ms=prefill_ms,
                decode_ms=decode_ms,
                load_ms=load_ms,
            )


        except Exception as e:
            # If anything goes wrong, return a failed result
            # instead of crashing the whole program
            latency_ms = (time.time() - start_time) * 1000
            return ModelResult(
                text="",
                model_name=self.model_name,
                tier=self.tier,
                latency_ms=latency_ms,
                success=False,
                error=str(e),
            )


class GeminiClient(BaseModelClient):
    """
    Client for Google Gemini API (cloud).

    HOW THE GEMINI API WORKS:
    You send an HTTP POST request to Google's servers with your prompt
    and API key. Google's powerful servers run the model and send back
    a response. You pay per token (but free tier is generous).

    WHY GEMINI:
    - Free tier: 60 requests/min for Flash, 2/min for Pro
    - Good quality across task types
    - Simple REST API (no SDK needed — fewer dependencies)
    """

    def __init__(self, model: str = "gemini-1.5-flash",
                 api_key: Optional[str] = None):
        self._model = model
        # Use provided key, or read from environment variable
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. Add it to your .env file."
            )
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"

    @property
    def model_name(self) -> str:
        return f"gemini/{self._model}"

    @property
    def tier(self) -> str:
        return "cloud"

    def call(self, prompt: str, system_prompt: Optional[str] = None,
             max_tokens: int = 1024, temperature: float = 0.7) -> ModelResult:

        start_time = time.time()

        try:
            # Gemini uses a "contents" array for conversation history
            # We simulate a system prompt by having a user/model exchange
            contents = []
            if system_prompt:
                contents.append({
                    "role": "user",
                    "parts": [{"text": system_prompt}]
                })
                contents.append({
                    "role": "model",
                    "parts": [{"text": "Understood. I will follow these instructions."}]
                })

            # Add the actual user prompt
            contents.append({
                "role": "user",
                "parts": [{"text": prompt}]
            })

            # Build the full request payload
            payload = {
                "contents": contents,
                "generationConfig": {
                    "maxOutputTokens": max_tokens,
                    "temperature": temperature,
                }
            }

            # The API key goes in the URL as a query parameter
            url = (f"{self.base_url}/models/{self._model}"
                   f":generateContent?key={self.api_key}")

            with httpx.Client(timeout=60.0) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()

            latency_ms = (time.time() - start_time) * 1000

            # Extract text from Gemini's response format
            # Gemini nests the response inside candidates → content → parts
            text = ""
            if "candidates" in data and len(data["candidates"]) > 0:
                parts = (data["candidates"][0]
                         .get("content", {})
                         .get("parts", []))
                text = "".join(p.get("text", "") for p in parts)

            # Extract token usage metadata
            usage = data.get("usageMetadata", {})

            return ModelResult(
                text=text,
                model_name=self.model_name,
                tier=self.tier,
                latency_ms=latency_ms,
                prompt_tokens=usage.get("promptTokenCount"),
                completion_tokens=usage.get("candidatesTokenCount"),
                total_tokens=usage.get("totalTokenCount"),
                raw_response=data,
                success=True,
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return ModelResult(
                text="",
                model_name=self.model_name,
                tier=self.tier,
                latency_ms=latency_ms,
                success=False,
                error=str(e),
            )


def get_available_models() -> Dict[str, BaseModelClient]:
    """
    Return all configured model clients.

    MODEL INVENTORY (as of your Ollama setup):

    EDGE (Local, $0.00):
      - gpt-oss:20b         → OpenAI's MoE model, strongest local reasoning,
                               but uses ~14GB RAM so it monopolizes your M1 Pro.
                               Has chain-of-thought and tool-calling built in.
      - llama3.1:8b          → Meta's instruction-tuned model, solid all-rounder.
      - deepseek-r1:8b       → Excellent at step-by-step reasoning, good for
                               analytical tasks.
      - qwen3:8b             → Alibaba's model, strong multilingual + general.
      - deepseek-coder:6.7b  → Specialized for code tasks, smaller and faster.

    CLOUD (Gemini API, per-token pricing):
      - gemini-2.5-flash      → Fast, cost-effective standard model.
      - gemini-2.5-flash-lite → Economy tier for high-volume tasks.
      - gemini-2.5-pro        → Premium reasoning and analysis.
      - gemini-3-pro-preview  → Cutting-edge frontier model.
    """
    models = {}

    # --- Edge models (local via Ollama) ---
    # Note: On 16GB M1 Pro, gpt-oss:20b uses ~14GB alone.
    # The 8B models use ~5-6GB each. Ollama swaps automatically.
    models["ollama/gpt-oss-20b"] = OllamaClient(model="gpt-oss:20b")
    models["ollama/llama3.1-8b"] = OllamaClient(
        model="llama3.1:8b-instruct-q5_K_M"
    )
    models["ollama/deepseek-r1-8b"] = OllamaClient(model="deepseek-r1:8b")
    models["ollama/qwen3-8b"] = OllamaClient(model="qwen3:8b")
    models["ollama/deepseek-coder-6.7b"] = OllamaClient(
        model="deepseek-coder:6.7b"
    )

    # --- Cloud models (only if API key is configured) ---
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        models["gemini/gemini-2.5-flash"] = GeminiClient(
            model="gemini-2.5-flash"
        )
        models["gemini/gemini-2.5-flash-lite"] = GeminiClient(
            model="gemini-2.5-flash-lite"
        )
        models["gemini/gemini-2.5-pro"] = GeminiClient(
            model="gemini-2.5-pro"
        )
        models["gemini/gemini-3-pro-preview"] = GeminiClient(
            model="gemini-3-pro-preview"
        )

    return models


def get_edge_models() -> Dict[str, BaseModelClient]:
    """
    Return ONLY edge (local) model clients.
    Used by the council — only local models participate in deliberation.

    WHY A SEPARATE FUNCTION?
    The council is made up exclusively of local models. We don't want
    cloud models voting on whether to use cloud — that's a conflict
    of interest (and would cost money just to vote).
    """
    all_models = get_available_models()
    return {
        name: client
        for name, client in all_models.items()
        if client.tier == "edge"
    }


def get_cloud_models() -> Dict[str, BaseModelClient]:
    """Return ONLY cloud model clients."""
    all_models = get_available_models()
    return {
        name: client
        for name, client in all_models.items()
        if client.tier == "cloud"
    }
