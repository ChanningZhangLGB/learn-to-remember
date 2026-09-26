"""The skill book: storage, embedding cache, meta updates, consolidation, pruning.

Two invariants the store owns and no LLM may touch:

  * `reliability` is DERIVED from helpful/harmful on every mutation. A value emitted by a
    model is ignored. In v0 it was initialized to 0.5 and never updated by anything.
  * Every mutation appends to a write log keyed by query id, so a reported gain can be
    audited entry by entry -- the L3 countermeasure from SPEC section 1.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .embed import DualEncoder
from .schema import EntryMeta, ProposedEntry, SkillEntry


@dataclass
class WriteLogRecord:
    step: int
    query_id: str
    action: str          # create | merge | reject | quarantine | prune | meta
    entry_id: str
    detail: str = ""


@dataclass
class SkillBook:
    encoder: DualEncoder
    entries: dict[str, SkillEntry] = field(default_factory=dict)
    write_log: list[WriteLogRecord] = field(default_factory=list)
    _next_id: int = 1
    _vectors: dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    _vector_version: int = field(default=-1, repr=False)

    # ------------------------------------------------------------------ basics

    def __len__(self) -> int:
        return len(self.entries)

    def active(self) -> list[SkillEntry]:
        return [e for e in self.entries.values() if e.status == "active"]

    def new_id(self) -> str:
        eid = f"m_{self._next_id:03d}"
        self._next_id += 1
        return eid

    def vector(self, entry: SkillEntry) -> np.ndarray:
        """Es(skill_view), cached. Invalidated on mutation and on any CCQS head update.

        The version check is the piece that keeps CCQS honest: once Es has taken a
        gradient step, every stored vector was produced by an older parameterization, and
        scoring a fresh query against them silently compares points from two different
        spaces. The frozen base embeddings underneath are *not* discarded, so the refresh
        is a matmul rather than a re-encode.
        """
        if self._vector_version != self.encoder.version:
            self._vectors.clear()
            self._vector_version = self.encoder.version
        vec = self._vectors.get(entry.id)
        if vec is None:
            vec = self.encoder.encode_skill([entry.skill_view()])[0]
            self._vectors[entry.id] = vec
        return vec

    def _invalidate(self, entry_id: str) -> None:
        self._vectors.pop(entry_id, None)

    def reset(self) -> None:
        """Empty the book completely. Called at the start of every run.

        `run.reset_per_run` used to reset only Ep/Es, so a caller that looped
        `pipeline.run(items)` to collect pass@1 over 5-10 passes silently carried the
        book from pass to pass -- run *n* answering items that runs 1..*n*-1 had already
        written entries about. That is exactly the prequential break the reset exists to
        prevent, and it failed silently: the accuracy simply drifted upward.

        The write log goes too. It is the per-run audit trail (SPEC L3); concatenating
        passes into it makes an entry's provenance unreadable.
        """
        self.entries.clear()
        self._vectors.clear()
        self.write_log.clear()
        self._next_id = 1
        self._vector_version = -1

    def log(self, step: int, query_id: str, action: str, entry_id: str,
            detail: str = "") -> None:
        self.write_log.append(WriteLogRecord(step, query_id, action, entry_id, detail))

    # ------------------------------------------------------------ mutation API

    def create(self, proposal: ProposedEntry, query_id: str, step: int,
               related_ids: list[str] | None = None) -> SkillEntry:
        entry = SkillEntry(
            id=self.new_id(),
            title=proposal.title,
            bullets=list(proposal.bullets),
            example=proposal.example,
            domain=proposal.domain,
            tags=list(proposal.tags),
            status="active",
            meta=EntryMeta(created=query_id, source_queries=[query_id],
                           related_ids=list(related_ids or [])),
        )
        entry.meta.recompute_reliability()
        self.entries[entry.id] = entry
        # Back-link so the relation is navigable from either side.
        for rid in entry.meta.related_ids:
            if rid in self.entries and entry.id not in self.entries[rid].meta.related_ids:
                self.entries[rid].meta.related_ids.append(entry.id)
        self.log(step, query_id, "create", entry.id, proposal.title)
        return entry

    def merge(self, target_id: str, proposal: ProposedEntry, query_id: str, step: int,
              max_bullets: int = 8) -> SkillEntry:
        """Fold a near-duplicate proposal into an existing entry.

        Title and id are kept for stability: retrieval statistics and the write log stay
        attached to a single identity across merges.
        """
        entry = self.entries[target_id]
        existing = {_norm_bullet(b) for b in entry.bullets}
        for b in proposal.bullets:
            if len(entry.bullets) >= max_bullets:
                break
            if _norm_bullet(b) not in existing:
                entry.bullets.append(b)
                existing.add(_norm_bullet(b))

        if len(proposal.example) > len(entry.example):
            entry.example = proposal.example
        for t in proposal.tags:
            if t not in entry.tags and len(entry.tags) < 4:
                entry.tags.append(t)

        if query_id not in entry.meta.source_queries:
            entry.meta.source_queries.append(query_id)
        entry.meta.cluster_size += 1
        self._invalidate(entry.id)
        self.log(step, query_id, "merge", entry.id,
                 f"cluster_size={entry.meta.cluster_size}")
        return entry

    def note_retrieved(self, entry_ids: list[str], step: int) -> None:
        for eid in entry_ids:
            e = self.entries.get(eid)
            if e is not None:
                e.meta.retrieved_count += 1
                e.meta.last_used_step = step

    def apply_evidence(self, entry_id: str, helpful: float = 0.0, harmful: float = 0.0,
                       query_id: str = "", step: int = 0) -> None:
        e = self.entries.get(entry_id)
        if e is None:
            return
        e.meta.helpful += max(0.0, helpful)
        e.meta.harmful += max(0.0, harmful)
        e.meta.recompute_reliability()
        self.log(step, query_id, "meta", entry_id,
                 f"h={e.meta.helpful:.2f} x={e.meta.harmful:.2f} r={e.meta.reliability:.3f}")

    # ------------------------------------------------- quarantine and pruning

    def quarantine_pass(self, cfg: dict, step: int, query_id: str) -> list[str]:
        """Withdraw entries whose evidence says they hurt. Retained, not deleted."""
        min_rel = float(cfg.get("quarantine_reliability", 0.25))
        min_ev = float(cfg.get("quarantine_min_evidence", 4.0))
        hit: list[str] = []
        for e in self.entries.values():
            if e.status != "active":
                continue
            if e.meta.evidence >= min_ev and e.meta.reliability < min_rel:
                e.status = "quarantined"
                hit.append(e.id)
                self.log(step, query_id, "quarantine", e.id,
                         f"r={e.meta.reliability:.3f} ev={e.meta.evidence:.1f}")
        return hit

    def prune_to_capacity(self, cfg: dict, step: int, query_id: str) -> list[str]:
        """Drop the weakest entries once the book exceeds capacity.

        Order: proven-bad first (has evidence, low reliability), then never-retrieved
        entries oldest-first. Entries with positive evidence are the last to go.
        """
        max_entries = int(cfg.get("max_entries", 512))
        actives = self.active()
        if len(actives) <= max_entries:
            return []

        def rank(e: SkillEntry) -> tuple[float, float, int]:
            never_used = 1.0 if e.meta.retrieved_count == 0 else 0.0
            return (e.meta.reliability, -never_used, e.meta.retrieved_count)

        actives.sort(key=rank)
        dropped: list[str] = []
        for e in actives[: len(actives) - max_entries]:
            del self.entries[e.id]
            self._invalidate(e.id)
            dropped.append(e.id)
            self.log(step, query_id, "prune", e.id, f"r={e.meta.reliability:.3f}")
        return dropped

    # ------------------------------------------------------------- persistence

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "next_id": self._next_id,
            "entries": [e.to_dict() for e in self.entries.values()],
            "write_log": [vars(r) for r in self.write_log],
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def load(path: str | Path, encoder: DualEncoder) -> "SkillBook":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        book = SkillBook(encoder=encoder)
        for d in payload.get("entries", []):
            e = SkillEntry.from_dict(d)
            e.meta.recompute_reliability()   # never trust a persisted derived value
            book.entries[e.id] = e
        book._next_id = int(payload.get("next_id", len(book.entries) + 1))
        book.write_log = [WriteLogRecord(**r) for r in payload.get("write_log", [])]
        return book

    def snapshot_stats(self) -> dict:
        actives = self.active()
        rels = [e.meta.reliability for e in actives]
        return {
            "size": len(self.entries),
            "active": len(actives),
            "quarantined": len(self.entries) - len(actives),
            "mean_reliability": float(np.mean(rels)) if rels else 0.0,
            "mean_cluster_size": (
                float(np.mean([e.meta.cluster_size for e in actives])) if actives else 0.0
            ),
        }


def _norm_bullet(text: str) -> str:
    return " ".join(text.lower().split())
