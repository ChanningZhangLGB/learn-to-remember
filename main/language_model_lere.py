import re
import tiktoken
from typing import List, Tuple, Optional
from .utils.execute_code import extract_and_run_python_code
from litellm import completion
from functools import partial


def _detect_and_truncate_repetition(text: str, max_repeats: int = 5) -> str:
    """
    Detect and truncate repetitive content in model output.

    If a substring of length >= 20 chars is repeated more than max_repeats
    consecutive times, truncate to keep only max_repeats occurrences and
    append a warning.

    Args:
        text: The model output text.
        max_repeats: Maximum allowed consecutive repetitions (default 5).

    Returns:
        str: Cleaned text with repetitions truncated.
    """
    if not text or len(text) < 100:
        return text

    # Strategy: find any substring (>=20 chars) that repeats consecutively
    # Use a regex to find long repeated patterns
    # Pattern: capture a group of 20+ chars that repeats 6+ times consecutively
    threshold = max_repeats + 1
    pattern = re.compile(r'(.{20,}?)\1{' + str(threshold - 1) + r',}', re.DOTALL)
    match = pattern.search(text)
    if match:
        repeated_unit = match.group(1)
        # Truncate: keep content up to the start of the repetition block,
        # then keep only max_repeats copies
        start = match.start()
        prefix = text[:start]
        kept = repeated_unit * max_repeats
        truncated = prefix + kept
        truncated += f"\n\n[WARNING: Repetitive output detected and truncated. " \
                     f"Repeated block ({len(repeated_unit)} chars) was found {match.group(0).count(repeated_unit)} times, " \
                     f"kept {max_repeats}.]"
        print(f"[RepetitionDetector] Truncated output: repeated block of {len(repeated_unit)} chars "
              f"found >={threshold} times at position {start}")
        return truncated

    # Also detect short repeated tokens/phrases (e.g., "= 8437.5 = 8437.5 = 8437.5...")
    # Find any token sequence repeated many times with a simpler pattern
    short_pattern = re.compile(r'(.{5,50}?)\1{' + str(threshold - 1) + r',}')
    match = short_pattern.search(text)
    if match:
        repeated_unit = match.group(1)
        start = match.start()
        prefix = text[:start]
        kept = repeated_unit * max_repeats
        truncated = prefix + kept
        truncated += f"\n\n[WARNING: Repetitive output detected and truncated. " \
                     f"Repeated block ({len(repeated_unit)} chars) was found >={threshold} times, " \
                     f"kept {max_repeats}.]"
        print(f"[RepetitionDetector] Truncated output: repeated block of {len(repeated_unit)} chars "
              f"found >={threshold} times at position {start}")
        return truncated

    return text

class LanguageModel:
    def __init__(self,
        model_name: str,
    ) -> None:
        """
        LanguageModel class to interact with different language models.

        Arguments:
            model_name : str : The name of the language model to use.
            api_key : str : The API key for the model.

        Raises:
            ValueError : If the model name is not found.
        """

        self.model_name = model_name

        # Load the client for the model based on the model name
        if self.model_name in [
            "openai/gpt-4o-mini", "openai/gpt-4o-mini-2024-07-18",
            "openai/gpt-4.1-mini", "openai/gpt-4.1-mini-2025-04-14", "openai/gpt-4.1-nano",
            "openai/gpt-4o", "openai/gpt-4o-2024-08-06", "openai/gpt-4o-2024-11-20",
            "openai/gpt-3.5-turbo",
            "openai/gpt-5-nano",
            "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "openai/o3-mini", "openai/o3-mini-2025-01-31",
            "openai/o1", "openai/o1-2024-12-17",
            "anthropic/claude-3-5-sonnet-latest", "anthropic/claude-3-5-sonnet-20241022",
            "anthropic/claude-3-5-haiku-latest", "anthropic/claude-3-5-haiku-20241022",
            "anthropic/claude-3-7-sonnet-latest", "anthropic/claude-3-7-sonnet-20250219",
            "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "together_ai/deepseek-ai/DeepSeek-R1",
            "together_ai/deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
            "together_ai/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            "together_ai/Qwen/Qwen2.5-Coder-32B-Instruct",
            "together_ai/Qwen/QwQ-32B",
            "together_ai/Qwen/Qwen2-72B-Instruct",
            "together_ai/Qwen/Qwen2.5-7B-Instruct-Turbo",
            "together_ai/Qwen/Qwen2.5-72B-Instruct-Turbo",
            "gemini/gemini-2.0-flash",
            "gemini/gemini-2.5-flash-lite",
            "gemini/gemini-2.5-flash-lite-preview-06-17",
            "gemini/gemini-2.5-flash",
            "gemini/gemini-2.5-pro",
            "ollama/llama3:70b",
            "ollama/qwen3:30b-instruct",
            "ollama/qwen3:30b",
            "ollama/deepseek-r1:32b",
            "ollama/gemma3:27b",
            "ollama/gpt-oss-safeguard:20b",
            "ollama/gpt-oss-safeguard:120b",
            "deepseek/deepseek-v3.1",
            "deepseek-chat",
            "deepseek-reasoner",
            "nvidia/deepseek-ai/deepseek-v3.1",
            "nvidia/qwen/qwen2.5-7b-instruct",
            "nvidia/qwen/qwen2.5-coder-32b-instruct",
            "nvidia/meta/llama-3.3-70b-instruct",
            "nvidia/openai/gpt-oss-20b",
            "nvidia/openai/gpt-oss-120b",
        ]:
            self.client = partial(completion, model=self.model_name)
        else:
            raise ValueError(f"Model '{model_name}' not found.")
        
        self.gpt4Tokenizer = tiktoken.encoding_for_model('gpt-4o')

    def get_embedding(self, text: str, model: str = "text-embedding-3-small") -> List[float]:
        """
        Get embedding for a text string using OpenAI's embedding API.

        Arguments:
            text: Text to embed
            model: Embedding model to use

        Returns:
            List of floats representing the embedding
        """
        from litellm import embedding
        # text-embedding-3-small has an 8192-token limit; truncate to stay safe
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            tokens = enc.encode(text)
            if len(tokens) > 8000:
                text = enc.decode(tokens[:8000])
        except Exception:
            if len(text) > 32000:
                text = text[:32000]
        response = embedding(model=f"openai/{model}", input=[text])
        return response.data[0]['embedding']


    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in the text.
        """
        tokens = self.gpt4Tokenizer.encode(text)
        return len(tokens)

    def generate(self,
        history: List[str], # cheatsheet history (for curation))
        temperature: float = 0.1,
        max_tokens: int = 2048,
        current_depth: int = 1,
        max_depth_num_rounds: int = 3,
        allow_code_execution: bool = True,
        code_execution_flag: str = "EXECUTE CODE!",
        final_output: str = ""
    ) -> str:
        """
        Generate a response from the language model.

        Arguments:
            history : List[str] : The conversation history.
            temperature : float : The sampling temperature for the model.
            max_tokens : int : The maximum number of tokens to generate.
            current_depth : int : The current depth of the conversation.
            max_depth_num_rounds : int : The maximum number of rounds allowed.
            allow_code_execution : bool : Whether to allow code execution.
            code_execution_flag : str : The flag to trigger code execution.
            final_output : str : The final output to return.

        Returns:
            str : The final output of the conversation.

        Raises:
            ValueError : If the history is empty.
        """
        if len(history) == 0:
            raise ValueError("History must contain at least one message.")
        

        # Generate the response from the language model
        # Special handling for gpt-5-nano (doesn't support temperature or max_completion_tokens)
        if self.model_name == "openai/gpt-5-nano":
            from openai import OpenAI
            import os
            openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = openai_client.chat.completions.create(
                model="gpt-5-nano",
                messages=history,
                # gpt-5-nano doesn't support max_completion_tokens or temperature
            )
            output = response.choices[0].message.content
        elif self.model_name == "nvidia/deepseek-ai/deepseek-v3.1":
            from openai import OpenAI
            import os, time
            nvidia_client = OpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=os.getenv("NVIDIA_API_KEY_DEEPSEEK"),
            )
            _max_retries = 5
            response = None
            for _attempt in range(1, _max_retries + 1):
                try:
                    response = nvidia_client.chat.completions.create(
                        model="deepseek-ai/deepseek-v3.1",
                        messages=history,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        extra_body={"chat_template_kwargs": {"thinking": True}},
                        stream=False,
                    )
                    break
                except Exception as _e:
                    if _attempt < _max_retries:
                        _wait = 15 * _attempt
                        print(f"[NVIDIA] Attempt {_attempt}/{_max_retries} failed ({type(_e).__name__}: {_e}). Retrying in {_wait}s...")
                        time.sleep(_wait)
                    else:
                        raise
            msg = response.choices[0].message
            output = msg.content or getattr(msg, "reasoning_content", "") or ""
        elif self.model_name == "nvidia/qwen/qwen2.5-7b-instruct":
            from openai import OpenAI
            import os, time
            nvidia_client = OpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=os.getenv("NVIDIA_API_KEY_QWEN"),
            )
            _max_retries = 5
            response = None
            for _attempt in range(1, _max_retries + 1):
                try:
                    response = nvidia_client.chat.completions.create(
                        model="qwen/qwen2.5-7b-instruct",
                        messages=history,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=False,
                    )
                    break
                except Exception as _e:
                    if _attempt < _max_retries:
                        _wait = 15 * _attempt
                        print(f"[NVIDIA-Qwen] Attempt {_attempt}/{_max_retries} failed ({type(_e).__name__}: {_e}). Retrying in {_wait}s...")
                        time.sleep(_wait)
                    else:
                        raise
            output = response.choices[0].message.content or ""
        elif self.model_name == "nvidia/qwen/qwen2.5-coder-32b-instruct":
            from openai import OpenAI
            import os, time
            nvidia_client = OpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=os.getenv("NVIDIA_API_KEY_QWEN"),
            )
            _max_retries = 5
            response = None
            for _attempt in range(1, _max_retries + 1):
                try:
                    response = nvidia_client.chat.completions.create(
                        model="qwen/qwen2.5-coder-32b-instruct",
                        messages=history,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=False,
                    )
                    break
                except Exception as _e:
                    if _attempt < _max_retries:
                        _wait = 15 * _attempt
                        print(f"[NVIDIA-QwenCoder32B] Attempt {_attempt}/{_max_retries} failed ({type(_e).__name__}: {_e}). Retrying in {_wait}s...")
                        time.sleep(_wait)
                    else:
                        raise
            output = response.choices[0].message.content or ""
        elif self.model_name == "nvidia/meta/llama-3.3-70b-instruct":
            from openai import OpenAI
            import os, time
            nvidia_client = OpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=os.getenv("NVIDIA_API_KEY_LLAMA"),
            )
            _max_retries = 5
            response = None
            for _attempt in range(1, _max_retries + 1):
                try:
                    response = nvidia_client.chat.completions.create(
                        model="meta/llama-3.3-70b-instruct",
                        messages=history,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=False,
                    )
                    break
                except Exception as _e:
                    if _attempt < _max_retries:
                        _wait = 15 * _attempt
                        print(f"[NVIDIA-Llama] Attempt {_attempt}/{_max_retries} failed ({type(_e).__name__}: {_e}). Retrying in {_wait}s...")
                        time.sleep(_wait)
                    else:
                        raise
            output = response.choices[0].message.content or ""
        elif self.model_name in ["nvidia/openai/gpt-oss-20b", "nvidia/openai/gpt-oss-120b"]:
            from openai import OpenAI
            import os, time
            api_key = os.getenv("NVIDIA_API_KEY_OSS_20B") if self.model_name == "nvidia/openai/gpt-oss-20b" else os.getenv("NVIDIA_API_KEY_OSS_120B")
            bare_name = "openai/gpt-oss-20b" if self.model_name == "nvidia/openai/gpt-oss-20b" else "openai/gpt-oss-120b"
            nvidia_client = OpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=api_key,
            )
            _max_retries = 5
            response = None
            for _attempt in range(1, _max_retries + 1):
                try:
                    response = nvidia_client.chat.completions.create(
                        model=bare_name,
                        messages=history,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=False,
                    )
                    break
                except Exception as _e:
                    if _attempt < _max_retries:
                        _wait = 15 * _attempt
                        print(f"[NVIDIA-OSS] Attempt {_attempt}/{_max_retries} failed ({type(_e).__name__}: {_e}). Retrying in {_wait}s...")
                        time.sleep(_wait)
                    else:
                        raise
            output = response.choices[0].message.content or ""
        elif self.model_name in ["deepseek-reasoner", "deepseek-chat", "deepseek/deepseek-v3.1"]:
            # DeepSeek API via OpenAI-compatible endpoint
            from openai import OpenAI
            import os
            base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
            api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
            openai_client = OpenAI(api_key=api_key, base_url=base_url)
            model_name = "deepseek-reasoner" if self.model_name == "deepseek/deepseek-v3.1" else self.model_name
            timeout_s = float(os.getenv("DEEPSEEK_TIMEOUT", "300"))
            max_retries = int(os.getenv("DEEPSEEK_MAX_RETRIES", "3"))
            ds_kwargs = {
                "model": model_name,
                "messages": history,
                "max_tokens": max_tokens,
                "timeout": timeout_s,
            }
            if model_name != "deepseek-reasoner":
                ds_kwargs["temperature"] = temperature
            response = None
            for attempt in range(1, max_retries + 1):
                try:
                    response = openai_client.chat.completions.create(**ds_kwargs)
                    break
                except Exception as exc:
                    import traceback, time
                    print(f"[DeepSeek] Attempt {attempt}/{max_retries} failed: {exc}")
                    traceback.print_exc()
                    if attempt < max_retries:
                        wait = 10 * attempt
                        print(f"[DeepSeek] Retrying in {wait}s...")
                        time.sleep(wait)
                    else:
                        raise
            message = response.choices[0].message
            output = message.content or getattr(message, "reasoning_content", "") or ""
        else:
            import time
            from litellm.exceptions import ServiceUnavailableError, RateLimitError
            _max_retries = 5
            for _attempt in range(1, _max_retries + 1):
                try:
                    output = self.client(
                        messages=history,
                        model=self.model_name,
                        temperature=temperature,
                        max_completion_tokens=max_tokens,
                    ).choices[0].message["content"]
                    break
                except (ServiceUnavailableError, RateLimitError) as _e:
                    if _attempt < _max_retries:
                        _wait = 15 * _attempt
                        print(f"[LiteLLM] Attempt {_attempt}/{_max_retries} failed ({type(_e).__name__}). Retrying in {_wait}s...")
                        time.sleep(_wait)
                    else:
                        raise

        # Repetition detection disabled
        # output = _detect_and_truncate_repetition(output, max_repeats=5)

        # Guard against None output (e.g. silent API failure)
        if output is None:
            output = ""

        # If Python code execution is allowed, execute the code
        pre_code_execution_flag = output.split(code_execution_flag)[0].strip()
        if allow_code_execution and code_execution_flag in output and '```' == pre_code_execution_flag[-3:]:
            if code_execution_flag in output:
                output_prefix = output.split(code_execution_flag)[0].strip()
            else:
                # TODO (msuzgun): This is a temporary solution. We may want to find a better way to handle this.
                output_prefix = output
            executed_code = extract_and_run_python_code(output_prefix)
            if executed_code is not None:
                executed_code = executed_code.strip()
            else:
                executed_code = ""
            current_output = f"{output_prefix}\n{code_execution_flag}\n\n{executed_code}"
            final_output = f"{final_output}\n\n{current_output}".strip()
            # import pdb; pdb.set_trase()
            # print(f"*** This code has been executed:\n{executed_code}\n\n")
            # print(f"***And the output is:\n{current_output}")
            # If the current depth is less than or equal to the maximum depth, add a new message to the history
            if current_depth <= max_depth_num_rounds:
                warning_txt = ""
                if current_depth == max_depth_num_rounds:
                    warning_txt = f" (This is the last round. No more code execution will be allowed. Please present your final solution now.)"
                new_messages = [
                    {"role": "assistant", "content": current_output},
                    {"role": "user", "content": f"Proceed with any additional steps required and provide the completed solution. If everything is already complete, type FINAL ANSWER and submit it in the expected format. If you are stucked, please try alternative methods to solve the problem and provide the final solution.{warning_txt}"}
                ]
                history += new_messages
                return self.generate(
                    history=history,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    current_depth=current_depth+1,
                    max_depth_num_rounds=max_depth_num_rounds,
                    allow_code_execution=allow_code_execution,
                    code_execution_flag=code_execution_flag,
                    final_output=final_output,
                )
            else:
                final_output = f"{final_output}\n\n{current_output}".strip()
                return final_output
        else:
            # If code execution is not allowed or no code block is found, return the final output
            # Add the output to the final output
            final_output = f"{final_output}\n\n{output}".strip()
            return final_output

