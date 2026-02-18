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

    WHY A UNIFORM RESULT?
    Whether you called Ollama locally or Gemini in the cloud,
    you get back this same structure. This makes logging,
    benchmarking, and routing all work the same way.
    No special cases, no "if cloud do X, if local do Y."
    """
    text: str  # The model's response
    model_name: str  # e.g., "ollama/mistral"
    tier: str  # "edge" or "cloud"
    latency_ms: float  # How long the call took
    prompt_tokens: Optional[int] = None  # Tokens in your prompt
    completion_tokens: Optional[int] = None  # Tokens in the response
    total_tokens: Optional[int] = None  # Sum of both
    raw_response: Optional[Dict[str, Any]] = None  # Full API response
    success: bool = True  # Did the call succeed?
    error: Optional[str] = None  # Error message if it failed


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

            return ModelResult(
                text=data.get("response", ""),
                model_name=self.model_name,
                tier=self.tier,
                latency_ms=latency_ms,
                prompt_tokens=data.get("prompt_eval_count"),
                completion_tokens=data.get("eval_count"),
                total_tokens=(
                        data.get("prompt_eval_count", 0) +
                        data.get("eval_count", 0)
                ),
                raw_response=data,
                success=True,
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

    This is the MODEL REGISTRY — a central place where all available
    models are instantiated. The router queries this to know what
    models it can use.
    """
    models = {}

    # Edge models (available if Ollama is running)
    models["ollama/llama3.1-8b"] = OllamaClient(model="llama3.1:8b-instruct-q5_K_M")
    models["ollama/deepseek-r1-8b"] = OllamaClient(model="deepseek-r1:8b")

    # Cloud models (only if API key is configured)
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
