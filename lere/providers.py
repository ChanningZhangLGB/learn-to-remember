"""Real provider client: one thin class over the OpenAI-compatible chat API.

`lere/llm.py` defines the whole interface -- `complete_json(prompt, component, image)`.
This module is the only implementation that talks to a network, and it is deliberately
one class rather than one class per vendor.

Why one class covers every target: gpt-4o-mini speaks `/chat/completions` natively,
Gemini exposes an OpenAI-compatible endpoint, and every open-weight serving stack worth
using (vLLM, Ollama, Together) does too. So `provider:` selects a base URL, a key name and
an image flavour, and nothing else. Adding a fourth model is a config entry.

What this deliberately does NOT do:

* **No native tool calling.** The Solver tool loop is prompt-side (`pipeline.solve`): the
  transcript is rendered into the next prompt and `complete_json` is called again. Native
  function calling differs per vendor and is exactly what would force this file to grow a
  branch per provider.
* **No SDK-level retries.** `max_retries=0` is passed to the client on purpose. The retry
  policy lives here so that backoff, jitter and the per-component retry counter are one
  implementation rather than two that disagree.

Cost is a headline metric and cannot be reconstructed after a run, so
every call is metered into `UsageLedger` keyed by component as it happens.
"""

from __future__ import annotations

import base64
import os
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from .llm import extract_json

REPO_ROOT = Path(__file__).resolve().parent.parent

COMPONENTS = ("planner", "solver", "curator")


# --------------------------------------------------------------------------- errors

class ProviderError(RuntimeError):
    """A provider failure that retrying will not fix: bad key, bad request, bad config."""


class TransientProviderError(ProviderError):
    """Rate limit, 5xx, timeout or dropped connection. Retried with backoff.

    A 12k-item MMLU-Pro run will hit all four; treating them as fatal would lose a run to
    a thirty-second outage.
    """

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class ParseFailure(ProviderError):
    """The model produced no recoverable JSON object after every parse retry.

    Raised rather than papered over: `pipeline.run` catches per-item exceptions and
    records them as error rows, so this costs one item, not the run.
    """


# ----------------------------------------------------------------------- key loading

# `API_key.txt` is a shell-style assignment file (`openAI="sk-..."`). Names are matched
# case-insensitively against the candidates in PROVIDERS below.
_ASSIGN_RE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")


def read_key_file(path: str | Path) -> dict:
    """Parse a `NAME=value` / `NAME="value"` file into a dict.

    Missing file returns {} rather than raising: the file is one of several places a key
    may live, and "not here" is not an error until every place has been tried.
    """
    p = Path(path)
    if not p.is_file():
        return {}
    out: dict = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _ASSIGN_RE.match(line)
        if not m:
            continue
        value = m.group(2)
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if value:
            out[m.group(1)] = value
    return out


def resolve_api_key(provider: str, *, key_file: str | Path | None = None,
                    env_var: str | None = None, env: dict | None = None) -> str:
    """Find the key. Never returns it in an error message.

    Order: the env var named in config, then the key file, then the provider's
    conventional env var names. The file is checked before the conventional names so that
    a key dropped in `API_key.txt` wins over a stale export in the shell.
    """
    env = os.environ if env is None else env
    spec = PROVIDERS.get(provider)
    if spec is None:
        raise ProviderError(
            "unknown provider %r; known: %s" % (provider, ", ".join(sorted(PROVIDERS))))

    if env_var:
        val = env.get(env_var)
        if val:
            return val.strip()

    path = Path(key_file) if key_file else (REPO_ROOT / "API_key.txt")
    entries = read_key_file(path)
    lowered = {k.lower(): v for k, v in entries.items()}
    for name in spec["key_names"]:
        if name.lower() in lowered:
            return lowered[name.lower()].strip()

    for name in spec["key_names"]:
        val = env.get(name)
        if val:
            return val.strip()

    raise ProviderError(
        "no API key for provider %r. Looked in: env %s, file %s (keys: %s), env names %s"
        % (provider, env_var or "(none configured)", path,
           ", ".join(sorted(entries)) or "none found",
           ", ".join(spec["key_names"])))


def scrub_secret(text: str, secret: str | None) -> str:
    """Remove the key from anything on its way to a log or an exception.

    `lere/tools.py` keeps the key out of the executor's environment deliberately; a
    traceback that echoes the request headers would put it straight back.
    """
    if not secret or not text:
        return text
    return text.replace(secret, "<redacted>")


# ------------------------------------------------------------------------- registries

PROVIDERS: dict = {
    "openai": {
        "base_url": None,                       # SDK default
        "key_names": ("openAI", "OPENAI_API_KEY", "openai"),
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "key_names": ("gemini", "GEMINI_API_KEY", "GOOGLE_API_KEY"),
    },
    "local": {                                  # vLLM / Ollama / any OpenAI-compatible
        "base_url": "http://localhost:11434/v1",
        "key_names": ("local", "LOCAL_API_KEY", "OPENAI_API_KEY"),
        "key_optional": True,
    },
    # NVIDIA NIM. OpenAI-compatible, but unlike `local` it DOES need a key, so it gets its
    # own entry rather than a base_url override on `local`: with key_optional the client
    # would send "not-needed" and every call would 403.
    "nvidia": {
        "base_url": "https://integrate.api.nvidia.com/v1",
        "key_names": ("nvidia", "NVIDIA_API_KEY"),
    },
}

# USD per million tokens. Checked 2026-09-04. Prices move, so `llm.price_per_mtok` in
# config overrides this and an unknown model costs 0.0 with `priced: false` in the ledger
# summary -- a visible zero rather than a wrong number.
PRICE_PER_MTOK: dict = {
    "gpt-4o-mini": {"input": 0.15, "cached_input": 0.075, "output": 0.60},
    "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
}


# ----------------------------------------------------------------------------- usage

@dataclass
class ComponentUsage:
    calls: int = 0
    prompt_tokens: int = 0
    cached_prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    retries: int = 0
    parse_failures: int = 0

    def as_dict(self) -> dict:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "cached_prompt_tokens": self.cached_prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "retries": self.retries,
            "parse_failures": self.parse_failures,
        }


class UsageLedger:
    """Tokens and dollars per component, accumulated as calls happen.

    Keyed by component because that is the unit the ablation is stated in: a small planner
    with a large solver is only interesting if the bill can be split Planner/Solver/Curator.
    """

    def __init__(self) -> None:
        self.by_component: dict = {}
        self.priced: bool = True

    def _slot(self, component: str) -> ComponentUsage:
        if component not in self.by_component:
            self.by_component[component] = ComponentUsage()
        return self.by_component[component]

    def record(self, component: str, *, prompt_tokens: int, cached_prompt_tokens: int,
               completion_tokens: int, cost_usd: float) -> None:
        slot = self._slot(component)
        slot.calls += 1
        slot.prompt_tokens += int(prompt_tokens)
        slot.cached_prompt_tokens += int(cached_prompt_tokens)
        slot.completion_tokens += int(completion_tokens)
        slot.cost_usd += float(cost_usd)

    def note_retry(self, component: str) -> None:
        self._slot(component).retries += 1

    def note_parse_failure(self, component: str) -> None:
        self._slot(component).parse_failures += 1

    @property
    def total_cost_usd(self) -> float:
        return sum(u.cost_usd for u in self.by_component.values())

    @property
    def total_calls(self) -> int:
        return sum(u.calls for u in self.by_component.values())

    def summary(self) -> dict:
        return {
            "calls": self.total_calls,
            "cost_usd": round(self.total_cost_usd, 6),
            "priced": self.priced,
            "prompt_tokens": sum(u.prompt_tokens for u in self.by_component.values()),
            "completion_tokens": sum(u.completion_tokens
                                     for u in self.by_component.values()),
            "by_component": {k: v.as_dict()
                             for k, v in sorted(self.by_component.items())},
        }


# ------------------------------------------------------------------------- transport

@dataclass
class TransportReply:
    text: str
    prompt_tokens: int = 0
    cached_prompt_tokens: int = 0
    completion_tokens: int = 0


class OpenAIChatTransport:
    """The one place an HTTP request is made. Injectable so tests never touch a network."""

    def __init__(self, api_key: str | None, *, base_url: str | None = None,
                 timeout_s: float = 120.0) -> None:
        from openai import OpenAI  # noqa: PLC0415 -- optional dependency, offline runs
        self._secret = api_key
        self._client = OpenAI(
            api_key=api_key or "not-needed",
            base_url=base_url,
            timeout=timeout_s,
            max_retries=0,      # retry policy lives in ProviderLLM; see module docstring
        )

    def __repr__(self) -> str:                       # never repr the key
        return "OpenAIChatTransport(base_url=%r)" % (str(self._client.base_url),)

    def send(self, *, model: str, messages: list, temperature: float,
             max_tokens: int, reasoning_effort: str | None = None,
             json_mode: bool = False) -> TransportReply:
        """One request. Every raise below is `from None` on purpose: the SDK exception
        carries the request object, and a chained traceback would print headers -- which
        is exactly where the key lives. The message survives, scrubbed.

        `max_tokens` is the classic field. Reasoning-family models want
        `max_completion_tokens` instead; none of the three targets does, and adding the
        branch before it is needed would be guessing at a model we have not chosen.
        """
        import openai  # noqa: PLC0415
        extra: dict = {}
        if reasoning_effort:
            extra["reasoning_effort"] = reasoning_effort
        if json_mode:
            extra["response_format"] = {"type": "json_object"}
        try:
            resp = self._client.chat.completions.create(
                model=model, messages=messages,
                temperature=temperature, max_tokens=max_tokens, **extra,
            )
        except (openai.RateLimitError, openai.APITimeoutError,
                openai.APIConnectionError, openai.InternalServerError) as exc:
            raise TransientProviderError(
                scrub_secret(str(exc), self._secret),
                retry_after=_retry_after(exc)) from None
        except openai.APIStatusError as exc:
            msg = scrub_secret(str(exc), self._secret)
            if exc.status_code == 429 or exc.status_code >= 500:
                raise TransientProviderError(msg, retry_after=_retry_after(exc)) from None
            raise ProviderError("HTTP %s: %s" % (exc.status_code, msg)) from None
        except Exception as exc:                     # unknown SDK failure
            raise ProviderError(scrub_secret(str(exc), self._secret)) from None

        choice = resp.choices[0] if resp.choices else None
        text = (getattr(choice.message, "content", None) or "") if choice else ""
        usage = getattr(resp, "usage", None)
        cached = 0
        details = getattr(usage, "prompt_tokens_details", None) if usage else None
        if details is not None:
            cached = int(getattr(details, "cached_tokens", 0) or 0)
        return TransportReply(
            text=text,
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0,
            cached_prompt_tokens=cached,
            completion_tokens=(int(getattr(usage, "completion_tokens", 0) or 0)
                               if usage else 0),
        )


def _retry_after(exc: object) -> float | None:
    """Honour the server's own backoff hint when it sends one."""
    resp = getattr(exc, "response", None)
    headers = getattr(resp, "headers", None)
    if not headers:
        return None
    for name in ("retry-after", "Retry-After", "x-ratelimit-reset-requests"):
        raw = headers.get(name) if hasattr(headers, "get") else None
        if raw:
            try:
                return max(0.0, float(str(raw).rstrip("s")))
            except ValueError:
                return None
    return None


# ------------------------------------------------------------------------ image parts

_MAGIC = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
)


def sniff_mime(data: bytes) -> str:
    """Identify the image from its magic bytes.

    Guessing a default would send MathVista's PNGs mislabelled and fail opaquely at the
    provider, or worse, be silently dropped and score the item on a question the model
    never saw.
    """
    for magic, mime in _MAGIC:
        if data.startswith(magic):
            return mime
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    raise ProviderError(
        "unrecognized image format (first bytes: %r); expected png, jpeg, gif or webp"
        % (data[:12],))


def image_data_uri(data: bytes) -> str:
    return "data:%s;base64,%s" % (sniff_mime(data),
                                  base64.b64encode(data).decode("ascii"))


# -------------------------------------------------------------------------- the client

@dataclass
class ComponentSpec:
    """Resolved per-component settings. `component` is the routing key, so Planner/Solver/Curator can
    sit on different models -- a small planner with a large solver, which the pipeline
    already routes."""
    model: str
    temperature: float
    max_tokens: int
    supports_images: bool
    price_input: float
    price_cached_input: float
    price_output: float
    priced: bool = True
    # Serving-side knobs. Both default to "absent" and are only forwarded when set, so a
    # transport that does not accept them (every fake in tests/) is untouched.
    reasoning_effort: str | None = None   # "none" turns a thinking model's reasoning off
    json_mode: bool = False               # response_format={"type": "json_object"}

    def cost(self, prompt_tokens: int, cached_prompt_tokens: int,
             completion_tokens: int) -> float:
        fresh = max(0, int(prompt_tokens) - int(cached_prompt_tokens))
        return (fresh / 1e6 * self.price_input
                + int(cached_prompt_tokens) / 1e6 * self.price_cached_input
                + int(completion_tokens) / 1e6 * self.price_output)


_CORRECTION = (
    "Your previous message was not parseable as a single JSON object. "
    "Reply with the JSON object that the instructions asked for and nothing else: "
    "no prose, no explanation, no markdown fence."
)


class ProviderLLM:
    """Satisfies the `LLM` protocol in `lere/llm.py` against a real provider."""

    def __init__(self, *, provider: str = "openai", model: str = "gpt-4o-mini",
                 temperature: float = 0.0, max_tokens: int = 4096,
                 supports_images: bool = True, reasoning_effort: str | None = None,
                 json_mode: bool = False, components: dict | None = None,
                 prices: dict | None = None, api_key: str | None = None,
                 base_url: str | None = None, timeout_s: float = 120.0,
                 max_retries: int = 5, backoff_base_s: float = 1.0,
                 backoff_max_s: float = 30.0, max_parse_retries: int = 2,
                 transport: object | None = None, sleep=time.sleep,
                 rng: random.Random | None = None, on_call=None) -> None:
        if provider not in PROVIDERS:
            raise ProviderError(
                "unknown provider %r; known: %s" % (provider, ", ".join(sorted(PROVIDERS))))
        self.provider = provider
        self.max_retries = int(max_retries)
        self.backoff_base_s = float(backoff_base_s)
        self.backoff_max_s = float(backoff_max_s)
        self.max_parse_retries = int(max_parse_retries)
        self.usage = UsageLedger()
        # Optional observer: (component, messages, raw_text, reply, parsed, attempts).
        # The raw text is otherwise discarded by `extract_json`, and it is the only place
        # a fence, a preamble or a near-miss schema violation is visible.
        self.on_call = on_call
        self._sleep = sleep
        self._rng = rng or random.Random(0)

        defaults = {"model": model, "temperature": temperature,
                    "max_tokens": max_tokens, "supports_images": supports_images,
                    "reasoning_effort": reasoning_effort, "json_mode": json_mode}
        self.specs: dict = {}
        for name in COMPONENTS:
            over = dict((components or {}).get(name) or {})
            merged = dict(defaults)
            merged.update(over)
            self.specs[name] = self._build_spec(merged, prices or {})
        if not all(s.priced for s in self.specs.values()):
            self.usage.priced = False

        if transport is None:
            base = base_url if base_url is not None else PROVIDERS[provider]["base_url"]
            transport = OpenAIChatTransport(api_key, base_url=base, timeout_s=timeout_s)
        self.transport = transport

    # ---------------------------------------------------------------- construction

    @staticmethod
    def _build_spec(cfg: dict, prices: dict) -> ComponentSpec:
        name = str(cfg["model"])
        table = dict(PRICE_PER_MTOK.get(name) or {})
        table.update(prices.get(name) or {})
        table.update(cfg.get("price_per_mtok") or {})
        priced = bool(table)
        p_in = float(table.get("input", 0.0))
        return ComponentSpec(
            model=name,
            temperature=float(cfg["temperature"]),
            max_tokens=int(cfg["max_tokens"]),
            supports_images=bool(cfg["supports_images"]),
            price_input=p_in,
            # Prompt caching triggers hard on a 12k-item run: every prompt shares a long
            # instruction prefix. Absent a cached price, cached tokens bill at full rate,
            # which overestimates rather than flattering the method.
            price_cached_input=float(table.get("cached_input", p_in)),
            price_output=float(table.get("output", 0.0)),
            priced=priced,
            reasoning_effort=(str(cfg["reasoning_effort"])
                              if cfg.get("reasoning_effort") else None),
            json_mode=bool(cfg.get("json_mode", False)),
        )

    @classmethod
    def from_config(cls, cfg: dict, *, transport: object | None = None,
                    api_key: str | None = None, env: dict | None = None,
                    on_call=None) -> "ProviderLLM":
        """Build from the FULL config dict, not just the `llm:` block.

        It needs `verification:` too, for the temperature cross-check below.
        """
        lcfg = dict(cfg.get("llm") or {})
        provider = str(lcfg.get("provider", "openai"))
        components = dict(lcfg.get("components") or {})

        if transport is None and api_key is None and not PROVIDERS.get(
                provider, {}).get("key_optional"):
            api_key = resolve_api_key(
                provider,
                key_file=lcfg.get("api_key_file"),
                env_var=lcfg.get("api_key_env"),
                env=env,
            )

        client = cls(
            provider=provider,
            model=str(lcfg.get("model", "gpt-4o-mini")),
            temperature=float(lcfg.get("temperature", 0.0)),
            max_tokens=int(lcfg.get("max_tokens", 4096)),
            supports_images=bool(lcfg.get("supports_images", True)),
            reasoning_effort=(str(lcfg["reasoning_effort"])
                              if lcfg.get("reasoning_effort") else None),
            json_mode=bool(lcfg.get("json_mode", False)),
            components=components,
            prices={str(lcfg.get("model", "gpt-4o-mini")):
                    dict(lcfg.get("price_per_mtok") or {})},
            api_key=api_key,
            base_url=lcfg.get("base_url"),
            timeout_s=float(lcfg.get("timeout_s", 120.0)),
            max_retries=int(lcfg.get("max_retries", 5)),
            backoff_base_s=float(lcfg.get("backoff_base_s", 1.0)),
            backoff_max_s=float(lcfg.get("backoff_max_s", 30.0)),
            max_parse_retries=int(lcfg.get("max_parse_retries", 2)),
            transport=transport,
            on_call=on_call,
        )
        client._check_consistency_temperature(cfg)
        return client

    def _check_consistency_temperature(self, cfg: dict) -> None:
        """Refuse a `consistency` arm that cannot produce a vote.

        `pipeline._consistency_samples` re-enters `solve()` with an identical prompt, and
        the client cannot tell a committed answer from a resample -- so temperature is a
        run-level setting. At temperature 0 the majority vote is five identical answers
        and the supervision signal is vacuous, silently. Fail at construction instead.

        Not detectable here: `source: gt` degrades to consistency per item when an item
        has no gold label. A gold-less dataset under `gt` needs the same temperature.
        """
        vcfg = cfg.get("verification") or {}
        samples = int(vcfg.get("consistency_samples", 5))
        if vcfg.get("source") == "consistency" and samples > 1:
            if self.specs["solver"].temperature <= 0.0:
                raise ProviderError(
                    "verification.source is 'consistency' with consistency_samples=%d, "
                    "but the Solver temperature is 0: every sample would be identical and "
                    "the majority vote meaningless. Set llm.components.solver.temperature "
                    "above 0." % samples)

    def __repr__(self) -> str:                       # never repr the key
        return "ProviderLLM(provider=%r, models=%r)" % (
            self.provider, {k: v.model for k, v in sorted(self.specs.items())})

    # --------------------------------------------------------------------- calling

    def complete_json(self, prompt: str, *, component: str,
                      image: bytes | None = None) -> dict:
        spec = self.specs.get(component)
        if spec is None:
            raise ProviderError("unknown component: %r" % (component,))
        messages = [{"role": "user", "content": self._content(prompt, image, spec,
                                                              component)}]

        last_error = ""
        for attempt in range(self.max_parse_retries + 1):
            reply = self._send(spec, messages, component)
            try:
                parsed = extract_json(reply.text)
            except ValueError as exc:                # json.JSONDecodeError subclasses it
                self.usage.note_parse_failure(component)
                last_error = str(exc)
                if attempt == self.max_parse_retries:
                    break
                # Resending the identical prompt at temperature 0 reproduces the identical
                # malformed output, so the retry has to change the conversation.
                messages = messages + [
                    {"role": "assistant", "content": (reply.text or "")[:2000]},
                    {"role": "user", "content": _CORRECTION},
                ]
                continue
            if isinstance(parsed, dict) and self.on_call is not None:
                self.on_call(component=component, messages=messages,
                             raw_text=reply.text, reply=reply, parsed=parsed,
                             attempts=attempt + 1, spec=spec)
            if not isinstance(parsed, dict):
                self.usage.note_parse_failure(component)
                last_error = "top-level JSON value is %s, not an object" % type(
                    parsed).__name__
                if attempt == self.max_parse_retries:
                    break
                messages = messages + [
                    {"role": "assistant", "content": (reply.text or "")[:2000]},
                    {"role": "user", "content": _CORRECTION},
                ]
                continue
            return parsed

        raise ParseFailure(
            "%s (%s) returned unparseable JSON after %d attempts: %s"
            % (component, spec.model, self.max_parse_retries + 1, last_error))

    def _content(self, prompt: str, image: bytes | None, spec: ComponentSpec,
                 component: str):
        if image is None:
            return prompt
        if not spec.supports_images:
            raise ProviderError(
                "component %s is configured on %r with supports_images=false but the item "
                "carries an image; it would be scored on a question the model never saw"
                % (component, spec.model))
        return [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_data_uri(image)}},
        ]

    def _send(self, spec: ComponentSpec, messages: list,
              component: str) -> TransportReply:
        for attempt in range(self.max_retries + 1):
            extra: dict = {}
            if spec.reasoning_effort:
                extra["reasoning_effort"] = spec.reasoning_effort
            if spec.json_mode:
                extra["json_mode"] = True
            try:
                reply = self.transport.send(
                    model=spec.model, messages=messages,
                    temperature=spec.temperature, max_tokens=spec.max_tokens, **extra)
            except TransientProviderError as exc:
                if attempt >= self.max_retries:
                    raise
                self.usage.note_retry(component)
                self._sleep(self._backoff(attempt, exc.retry_after))
                continue
            self.usage.record(
                component,
                prompt_tokens=reply.prompt_tokens,
                cached_prompt_tokens=reply.cached_prompt_tokens,
                completion_tokens=reply.completion_tokens,
                cost_usd=spec.cost(reply.prompt_tokens, reply.cached_prompt_tokens,
                                   reply.completion_tokens))
            return reply
        raise ProviderError("unreachable: retry loop exited without a reply")

    def _backoff(self, attempt: int, retry_after: float | None) -> float:
        if retry_after is not None:
            # Honour the server's number, plus a little jitter so a batch of parallel
            # workers does not synchronize on the same reset instant.
            return retry_after + self._rng.random() * 0.5
        capped = min(self.backoff_max_s, self.backoff_base_s * (2 ** attempt))
        return capped * (0.5 + self._rng.random() * 0.5)
