"""
Reference snippets for calling open-source LLMs through the NVIDIA NIM
OpenAI-compatible endpoint. These are **not** imported by the pipeline —
the pipeline uses `litellm` via `model_call` in `language_model_lere.py`.

Set `NVIDIA_API_KEY_*` in your `config.env` before running any of the
examples below.
"""

import os
from openai import OpenAI


def deepseek_example(prompt: str = "") -> None:
    """DeepSeek V3.1 (reasoning mode) via NVIDIA NIM."""
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.environ["NVIDIA_API_KEY_DEEPSEEK"],
    )
    completion = client.chat.completions.create(
        model="deepseek-ai/deepseek-v3.1",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        top_p=0.7,
        max_tokens=8192,
        extra_body={"chat_template_kwargs": {"thinking": True}},
        stream=True,
    )
    for chunk in completion:
        if not getattr(chunk, "choices", None):
            continue
        reasoning = getattr(chunk.choices[0].delta, "reasoning_content", None)
        if reasoning:
            print(reasoning, end="")
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="")


def qwen_example(prompt: str = "") -> None:
    """Qwen 2.5 Coder 32B via NVIDIA NIM."""
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.environ["NVIDIA_API_KEY_QWEN"],
    )
    completion = client.chat.completions.create(
        model="qwen/qwen2.5-coder-32b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        top_p=0.7,
        max_tokens=1024,
        stream=True,
    )
    for chunk in completion:
        if chunk.choices and chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="")


def llama_example(prompt: str = "") -> None:
    """Llama 3.3 70B Instruct via NVIDIA NIM."""
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.environ["NVIDIA_API_KEY_LLAMA"],
    )
    completion = client.chat.completions.create(
        model="meta/llama-3.3-70b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        top_p=0.7,
        max_tokens=1024,
        stream=True,
    )
    for chunk in completion:
        if chunk.choices and chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="")
