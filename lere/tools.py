"""Code execution for the solver's tool calls.

Solver may write a program and reason from what it actually printed. Before this module
existed, Solver's prompt asked it to "reason from its result" and emit `coding_result` -- with
nothing running the code. The model invented its own program's output, and that invention
was then shown to Curator as though it were execution evidence. Every number downstream of that
was a hallucination wearing the costume of a measurement.

Two things this module is, and one it is not:

  * It is the **reasoning-time** executor. The result goes back into Solver's context and Solver
    continues from it. That is different from `verify.signal_from_exec`, which runs
    *after* an answer exists in order to check it. Both exist; they are not the same tool
    and must not be collapsed.
  * It is a **speed bump**, not a sandbox. The import denylist and the scrubbed
    environment stop an unlucky model, not a determined one. The real boundary is the
    machine: run experiments in a VM or container with no network and nothing valuable
    mounted. This is stated again in README.md because it is the one place where
    getting it wrong is expensive.

The environment scrub is not decoration. The parent process holds the provider API key
that pays for the run, and the child is executing text a language model wrote. The child
gets the minimum set of variables an interpreter needs to start, and nothing else.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

# Import roots refused before execution. Network and process-spawning are the ones that
# matter: a solver that reaches the internet mid-run makes the run unreproducible, and one
# that spawns processes escapes the timeout that bounds it.
DEFAULT_DENY_IMPORTS = (
    "socket", "urllib", "urllib2", "urllib3", "requests", "httpx", "http", "ftplib",
    "telnetlib", "smtplib", "xmlrpc", "asyncio", "subprocess", "multiprocessing",
    "ctypes", "shutil", "pty", "webbrowser", "pickle", "marshal", "importlib",
)

# Variables the child keeps. Everything else -- API keys above all -- is dropped.
_ENV_KEEP = (
    "SYSTEMROOT", "WINDIR", "PATH", "PATHEXT", "COMSPEC", "TEMP", "TMP", "TMPDIR",
    "HOME", "LANG", "LC_ALL", "NUMBER_OF_PROCESSORS",
)


@dataclass
class ToolResult:
    """One execution. `ok` means the program ran and exited zero, nothing more."""

    ok: bool
    stdout: str = ""
    stderr: str = ""
    duration_s: float = 0.0
    truncated: bool = False
    error: str = ""          # syntax_error | denied_import:<mod> | timeout | exit_<n>
    code: str = ""
    repeated: bool = False   # this program had already been run this item; not re-executed

    def render(self) -> str:
        """How the result is shown back to Solver. Failures are shown, never swallowed.

        A traceback is the most useful thing a solver can receive -- it is how a human
        debugs, and hiding it behind "execution failed" throws away the whole benefit of
        having run the code at all.
        """
        head = "TOOL RESULT (python)" if self.ok else f"TOOL ERROR (python) [{self.error}]"
        parts = [head]
        if self.stdout:
            parts.append(f"stdout:\n{self.stdout}")
        if self.stderr:
            parts.append(f"stderr:\n{self.stderr}")
        if not self.stdout and not self.stderr:
            parts.append("(the program printed nothing -- if you expected output, "
                         "add a print() and try again)")
        if self.truncated:
            parts.append("(output truncated)")
        return "\n".join(parts)


@dataclass
class ToolConfig:
    enabled: bool = True
    max_calls_per_item: int = 3
    timeout_s: int = 20
    max_output_chars: int = 4000
    # Address-space cap (POSIX only). 512 MB was too small for numpy: instead of raising
    # MemoryError the allocator thrashes and the import never returns, so the run burns
    # the whole timeout AND an LLM turn on a library the prompt told the solver to use.
    # numpy imports in 0.2s at 1024; 2048 leaves headroom for the arrays it then makes.
    # `null` disables the cap entirely.
    memory_mb: int | None = 2048
    deny_imports: tuple[str, ...] = DEFAULT_DENY_IMPORTS

    @staticmethod
    def from_cfg(cfg: dict | None) -> "ToolConfig":
        cfg = cfg or {}
        deny = cfg.get("deny_imports")
        return ToolConfig(
            enabled=bool(cfg.get("enabled", True)),
            max_calls_per_item=int(cfg.get("max_calls_per_item", 3)),
            timeout_s=int(cfg.get("timeout_s", 20)),
            max_output_chars=int(cfg.get("max_output_chars", 4000)),
            memory_mb=(None if cfg.get("memory_mb", 2048) is None
                       else int(cfg.get("memory_mb", 2048))),
            deny_imports=tuple(deny) if deny else DEFAULT_DENY_IMPORTS,
        )


# Offered to the solver if and only if they actually import. `sympy` is listed because it
# is the one worth having on AIME; when it is absent it simply does not appear.
CANDIDATE_MODULES = (
    "math", "itertools", "fractions", "decimal", "collections", "functools",
    "statistics", "re", "random", "numpy", "sympy",
)

_PROBE_CACHE: dict = {}


def probe_modules(cfg: "ToolConfig",
                  candidates: tuple = CANDIDATE_MODULES) -> list:
    """Which of `candidates` import successfully *through the executor*.

    The prompt used to hard-code this list and it drifted: it advertised `sympy`, which is
    not installed, and `numpy`, which the old memory cap made hang. Every one of the 20
    tool failures in the last run was the solver believing that list. Probing through
    `run_python` rather than in-process means the answer reflects the isolated interpreter,
    the import denylist and the resource limits the solver will actually face.
    """
    key = (tuple(candidates), cfg.memory_mb, cfg.timeout_s)
    if key in _PROBE_CACHE:
        return _PROBE_CACHE[key]
    ok = []
    for name in candidates:
        r = run_python("import %s\nprint('ok')" % name, cfg)
        if r.ok and r.stdout.strip() == "ok":
            ok.append(name)
    _PROBE_CACHE[key] = ok
    return ok


def _imported_roots(tree: ast.AST) -> set[str]:
    """Top-level module names the source imports, by AST rather than by regex.

    Regex on import lines misses `from a.b import c` and fires on the word "socket" in a
    comment. The AST is exact for static imports, which is the whole class this check is
    meant to cover -- `__import__` is handled separately.
    """
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                roots.add(node.module.split(".", 1)[0])
        elif isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name) and fn.id == "__import__":
                # A dynamic import defeats the static check; refuse the whole program
                # rather than pretend the check covered it.
                roots.add("__import__")
    return roots


def static_check(code: str, cfg: ToolConfig) -> str:
    """Return an error tag if the code must not run, or "" to allow it.

    Runs before the subprocess so a refusal costs nothing and, more usefully, so the
    reason can be handed back to Solver as feedback it can act on.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"syntax_error: line {exc.lineno}: {exc.msg}"
    denied = _imported_roots(tree) & set(cfg.deny_imports)
    if "__import__" in _imported_roots(tree):
        denied.add("__import__")
    if denied:
        return f"denied_import:{','.join(sorted(denied))}"
    return ""


def _child_env() -> dict:
    env = {k: os.environ[k] for k in _ENV_KEEP if k in os.environ}
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONHASHSEED"] = "0"          # one less source of run-to-run drift
    return env


def _limit_memory(mb: int):
    """POSIX address-space cap, as a preexec_fn. Returns None where unavailable.

    Windows has no `resource`, so on Windows the timeout is the only bound and a program
    that allocates without pause will swap before it is killed. Linux VM is the intended
    venue for real runs, and this is one of the reasons.
    """
    try:
        import resource                                     # noqa: PLC0415
    except ImportError:
        return None

    def _apply() -> None:
        limit = mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

    return _apply


def _clip(text: str, limit: int) -> tuple[str, bool]:
    text = text or ""
    if len(text) <= limit:
        return text.strip(), False
    return text[:limit].strip(), True


def run_python(code: str, cfg: ToolConfig | None = None) -> ToolResult:
    """Execute one program and capture what it printed.

    Isolated interpreter (`-I`: no user site-packages, no PYTHON* env influence), a
    throwaway working directory, a scrubbed environment, a wall-clock timeout, and an
    address-space cap where the platform provides one.
    """
    cfg = cfg or ToolConfig()
    code = (code or "").strip()
    if not code or code == "N/A":
        return ToolResult(ok=False, error="empty_code", code=code)

    problem = static_check(code, cfg)
    if problem:
        return ToolResult(ok=False, error=problem, code=code,
                          stderr=f"refused before execution: {problem}")

    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as td:
        script = Path(td) / "solve.py"
        script.write_text(code, encoding="utf-8")
        kwargs: dict = {}
        preexec = _limit_memory(cfg.memory_mb)
        if preexec is not None:
            kwargs["preexec_fn"] = preexec
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(script)],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=cfg.timeout_s, cwd=td, env=_child_env(), **kwargs,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(ok=False, error="timeout", code=code,
                              duration_s=time.perf_counter() - t0,
                              stderr=f"no result within {cfg.timeout_s}s")
        except OSError as exc:
            return ToolResult(ok=False, error="spawn_failed", code=code,
                              duration_s=time.perf_counter() - t0, stderr=str(exc))

    elapsed = time.perf_counter() - t0
    out, out_cut = _clip(proc.stdout, cfg.max_output_chars)
    err, err_cut = _clip(proc.stderr, cfg.max_output_chars)
    return ToolResult(
        ok=(proc.returncode == 0),
        stdout=out, stderr=err, duration_s=elapsed,
        truncated=(out_cut or err_cut),
        error="" if proc.returncode == 0 else f"exit_{proc.returncode}",
        code=code,
    )


@dataclass
class ToolTranscript:
    """The running record of one item's tool use.

    Held separately from `SolverOutput` because it is evidence about the run rather than
    something the model said: `SolverOutput.coding_result` is overwritten from here, so a
    model that misreports what its own program printed cannot mislead Curator.
    """

    results: list[ToolResult] = field(default_factory=list)
    repeats: int = 0

    def add(self, result: ToolResult) -> None:
        self.results.append(result)

    @staticmethod
    def _key(code: str) -> str:
        """Identity for a program, up to indentation and line breaks.

        Deliberately conservative: it will not see `print( 0 )` and `print(0)` as the same
        program. Under-detecting costs one redundant subprocess; over-detecting would
        withhold a genuinely different program's output from the solver, which is the far
        worse failure. The case this exists for is byte-identical re-emission.
        """
        return " ".join((code or "").split())

    def find_repeat(self, code: str) -> "ToolResult | None":
        """A previous run of the identical program, if there is one.

        A model that re-emits byte-identical code is not learning anything new from the
        second run: same program, same throwaway directory, same output. Re-executing it
        costs a subprocess and, worse, hides the repetition inside a transcript that looks
        like three independent investigations.
        """
        key = self._key(code)
        if not key:
            return None
        for r in self.results:
            if self._key(r.code) == key:
                return r
        return None

    def add_repeat(self, prior: ToolResult) -> ToolResult:
        """Record a re-run without spawning a second subprocess.

        It still counts towards `calls`: the LLM turn was spent either way, and the budget
        is what eventually forces the model to commit. Not counting it would let a model
        loop on one program forever.
        """
        echoed = replace(prior, repeated=True)
        self.results.append(echoed)
        self.repeats += 1
        return echoed

    @property
    def calls(self) -> int:
        return len(self.results)

    @property
    def failures(self) -> int:
        return sum(1 for r in self.results if not r.ok)

    @property
    def successes(self) -> list:
        """Every run that executed and exited zero, in order.

        Verification needs the whole list, not just the last one: `last_success` is the
        answer-producing program only by coincidence, and two successful runs printing
        different values means the source cannot say which one the answer came from.
        """
        return [r for r in self.results if r.ok]

    @property
    def last_success(self) -> ToolResult | None:
        for r in reversed(self.results):
            if r.ok:
                return r
        return None

    def render(self) -> str:
        """The transcript as Solver sees it on its next turn."""
        if not self.results:
            return ""
        blocks = []
        for i, r in enumerate(self.results, 1):
            note = ("\n[NOT RE-RUN: this is byte-for-byte the program you already ran "
                    "above. The output below is that run's output. Running it again "
                    "cannot change it.]" if r.repeated else "")
            blocks.append(f"--- call {i} ---\ncode:\n{r.code}{note}\n\n{r.render()}")
        return "\n\n".join(blocks)
