"""Encoders: a frozen base plus the two trainable projection heads E_q and E_m.

An earlier design enforced a single shared encoder object. That was too strong. The real
constraint is that the query side and the skill side must be **trained jointly by one
loss** -- two *independently* trained encoders put their outputs in unaligned subspaces
where cosine similarity carries no usable signal. A dual head over one frozen base,
coupled by L_ccme, satisfies that and is what CCME needs (see `lere/ccme.py`).

Two properties are load-bearing:

  * **Identity initialization.** Both heads start as (a truncated/padded) identity, so an
    untrained CCME reproduces frozen-base retrieval exactly. This makes the
    `ccme.enabled: false` ablation an exact control, and it removes the cold-start regime
    where a randomly projected space would be worse than no projection at all.
  * **Base embeddings are cached, heads are not.** A head update invalidates only the
    projected vectors; the expensive base pass is never repeated. Refreshing the whole
    memory after an update is one (n x d) @ (d x p) matmul.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol, Sequence

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class Encoder(Protocol):
    dim: int

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return an (n, dim) L2-normalized float32 array."""
        ...


def l2_normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (m / norms).astype(np.float32)


class HashingEncoder:
    """Dependency-free deterministic encoder: hashed token bigrams with sublinear tf.

    Not competitive with a trained sentence encoder -- it exists so the pipeline, the
    tests, and the threshold-calibration script all run with no model download and no
    network. Real runs use SentenceTransformerEncoder. Similarity scales differ
    substantially between the two, so `sim_threshold` must be recalibrated when switching
    (see README, Calibration).
    """

    def __init__(self, dim: int = 512, seed: int = 0) -> None:
        self.dim = dim
        self.seed = seed

    def _bucket(self, token: str) -> int:
        h = hashlib.blake2b(f"{self.seed}:{token}".encode(), digest_size=8).digest()
        return int.from_bytes(h, "big") % self.dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            toks = _tokens(text)
            grams = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
            counts: dict[int, float] = {}
            for g in grams:
                b = self._bucket(g)
                counts[b] = counts.get(b, 0.0) + 1.0
            for b, c in counts.items():
                out[i, b] = 1.0 + np.log(c)   # sublinear tf damps repeated boilerplate
        return l2_normalize(out)


class SentenceTransformerEncoder:
    """sentence-transformers wrapper. Import is lazy so the package stays optional.

    `all-MiniLM-L6-v2` (384-d) is the configured default: local, free, and cheap enough
    that re-encoding the memory on every CCME update is not a cost consideration. It is
    text-only, so multimodal items reach it through the planner's `visual_context`.
    """

    def __init__(self, model: str = "all-MiniLM-L6-v2", device: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        self._model = SentenceTransformer(model, device=device)
        # renamed in sentence-transformers 5.x; keep both so the pin stays loose
        getter = getattr(self._model, "get_embedding_dimension", None)
        if getter is None:
            getter = self._model.get_sentence_embedding_dimension
        self.dim = int(getter())

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vecs = self._model.encode(
            list(texts), convert_to_numpy=True, show_progress_bar=False,
            normalize_embeddings=False,
        )
        return l2_normalize(np.asarray(vecs, dtype=np.float32))


# ------------------------------------------------------------------ projection heads

def _identity_init(in_dim: int, out_dim: int) -> np.ndarray:
    """(out_dim, in_dim) truncated/padded identity.

    At out_dim <= in_dim this keeps the first `out_dim` coordinates, which for a
    normalized sentence embedding preserves most of the pairwise geometry; the point is
    not that it is optimal but that it is a *fixed, non-random* starting point, so the
    trained and untrained arms differ only by what training did.
    """
    w = np.zeros((out_dim, in_dim), dtype=np.float32)
    for i in range(min(in_dim, out_dim)):
        w[i, i] = 1.0
    return w


class ProjectionHead:
    """One linear head, identity-initialized. Wraps torch when available.

    Kept deliberately small: with an empty memory at step 0 and a few hundred queries in a
    whole AIME run, the realistic yield is on the order of 10^2 positive pairs. A single
    linear map with weight decay is roughly the largest thing that data can support.
    """

    def __init__(self, in_dim: int, out_dim: int, name: str = "head") -> None:
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.name = name
        self._torch = None
        self._module = None
        self._w = _identity_init(in_dim, out_dim)
        self._try_torch()

    def _try_torch(self) -> None:
        try:
            import torch                                   # noqa: PLC0415
            from torch import nn                           # noqa: PLC0415
        except ImportError:
            return
        self._torch = torch
        mod = nn.Linear(self.in_dim, self.out_dim, bias=False)
        with torch.no_grad():
            mod.weight.copy_(torch.from_numpy(self._w))
        self._module = mod

    @property
    def trainable(self) -> bool:
        return self._module is not None

    @property
    def module(self):
        """The torch module, for the optimizer. None when torch is unavailable."""
        return self._module

    def forward_torch(self, x):
        """Differentiable path used by the CCME trainer. L2-normalized output."""
        t = self._torch
        return t.nn.functional.normalize(self._module(x), dim=-1)

    def apply(self, base: np.ndarray) -> np.ndarray:
        """Inference path: (n, in_dim) base vectors -> (n, out_dim) L2-normalized."""
        if base.size == 0:
            return np.zeros((0, self.out_dim), dtype=np.float32)
        if self._module is not None:
            t = self._torch
            with t.no_grad():
                out = self._module(t.from_numpy(np.ascontiguousarray(base))).numpy()
        else:
            out = base @ self._w.T
        return l2_normalize(out)

    def reset(self) -> None:
        """Back to identity. Called at the start of every run."""
        self._w = _identity_init(self.in_dim, self.out_dim)
        if self._module is not None:
            with self._torch.no_grad():
                self._module.weight.copy_(self._torch.from_numpy(self._w))


class DualEncoder:
    """E_q (queries) and E_m (skill entries) over one frozen base.

    `version` increments on every head update. The memory watches it and drops its projected
    vector cache, which is the mechanism that keeps stored vectors from going stale after
    CCME learns something -- a silent failure otherwise, since retrieval would keep
    scoring against embeddings from an older parameterization.
    """

    def __init__(self, base: Encoder, proj_dim: int | None = None) -> None:
        self._base = base
        self.base_dim = base.dim
        self.dim = int(proj_dim or base.dim)
        self.eq = ProjectionHead(self.base_dim, self.dim, "E_q")
        self.em = ProjectionHead(self.base_dim, self.dim, "E_m")
        self._base_cache: dict[str, np.ndarray] = {}
        self.version = 0

    @property
    def base(self) -> Encoder:
        return self._base

    @property
    def trainable(self) -> bool:
        return self.eq.trainable and self.em.trainable

    # ------------------------------------------------------------ base caching

    def base_vectors(self, texts: Sequence[str]) -> np.ndarray:
        """Frozen base embeddings, cached across the whole run. Never invalidated."""
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.base_dim), dtype=np.float32)
        missing = [t for t in texts if t not in self._base_cache]
        if missing:
            uniq = list(dict.fromkeys(missing))       # order-preserving de-dup
            for t, v in zip(uniq, self._base.encode(uniq)):
                self._base_cache[t] = v
        return np.stack([self._base_cache[t] for t in texts])

    # ------------------------------------------------------------- projections

    def encode_query(self, text: str) -> np.ndarray:
        """E_q."""
        return self.eq.apply(self.base_vectors([text]))[0]

    def encode_entries(self, texts: Sequence[str]) -> np.ndarray:
        """E_m."""
        return self.em.apply(self.base_vectors(list(texts)))

    def bump_version(self) -> int:
        self.version += 1
        return self.version

    def reset_heads(self) -> None:
        """Re-initialize E_q and E_m to identity.

        Called at the start of every run. Carrying heads across runs would break the
        prequential guarantee: run n would be answering items whose labels shaped the
        encoder in runs 1..n-1, so an item would no longer be predicted before its own
        label was used.
        """
        self.eq.reset()
        self.em.reset()
        self.bump_version()


def build_encoder(cfg: dict) -> DualEncoder:
    cfg = cfg or {}
    backend = cfg.get("backend", "hashing")
    if backend == "hashing":
        base: Encoder = HashingEncoder(dim=int(cfg.get("dim", 512)))
    elif backend == "sentence_transformers":
        base = SentenceTransformerEncoder(cfg.get("model", "all-MiniLM-L6-v2"))
    else:
        raise ValueError(f"unknown embedding backend: {backend!r}")
    return DualEncoder(base, proj_dim=cfg.get("proj_dim"))
