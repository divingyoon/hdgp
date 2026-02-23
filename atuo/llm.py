from __future__ import annotations

import json
import os
import urllib.request


def call_openai_chat(prompt: str, model: str, temperature: float, max_tokens: int, api_base: str) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    url = api_base.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an RL reward engineering expert analyzing bimanual grasping tasks. Analyze the training data step-by-step, then output JSON with keys: analysis, overrides. The 'analysis' field should contain your step-by-step reasoning. The 'overrides' field should be a list of 'key=value' strings."},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=60) as resp:
        body = resp.read().decode("utf-8")

    data = json.loads(body)
    choices = data.get("choices", [])
    if not choices:
        raise RuntimeError("OpenAI response missing choices")
    msg = choices[0].get("message", {})
    content = msg.get("content", "")
    if not content:
        raise RuntimeError("OpenAI response missing content")
    return content


def call_ollama_chat(prompt: str, model: str, temperature: float, api_base: str) -> str:
    return call_ollama_messages(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an RL reward engineering expert analyzing bimanual grasping tasks. "
                    "Analyze the training data step-by-step, then output JSON with keys: analysis, overrides. "
                    "The 'analysis' field should contain your step-by-step reasoning. "
                    "The 'overrides' field should be a list of 'key=value' strings."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        model=model,
        temperature=temperature,
        api_base=api_base,
    )


def call_ollama_messages(messages: list[dict], model: str, temperature: float, api_base: str) -> str:
    url = api_base.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "options": {"temperature": temperature},
        "stream": False,
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=60) as resp:
        body = resp.read().decode("utf-8")

    data = json.loads(body)
    message = data.get("message", {})
    content = message.get("content", "")
    if not content:
        raise RuntimeError("Ollama response missing content")
    return content
