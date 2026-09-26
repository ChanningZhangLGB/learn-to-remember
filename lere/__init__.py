"""LeRe: Learn to Remember -- geometric memory for inference-time self-improvement.

Code names vs. paper notation:
  c1 / c2 / c3 (PlannerOutput, SolverOutput, CuratorOutput)   Planner P / Solver G / Curator C
  SkillBook, SkillEntry, "note"                               memory bank M, memory entry m
  CCQS (CCQSTrainer, L_ccqs), heads Ep / Es                   CCME (L_CCME), heads E_q / E_m
  helpful / harmful evidence, reliability                     h_m / f_m, p_hat(m)  (Eq. 8)
  guard / curate (merge, link) / pruning (quarantine)         GCM: Guard / Consolidate / Maintain
"""

from .schema import SkillEntry, EntryMeta, PlannerOutput, SolverOutput, CuratorOutput
from .embed import DualEncoder, HashingEncoder, ProjectionHead, build_encoder
from .ccqs import CCQSTrainer, CCQSPair
from .store import SkillBook
from .retrieve import Retriever
from .verify import VerificationSignal, build_signal
from .tools import ToolConfig, ToolResult, ToolTranscript, run_python
from .pipeline import Pipeline, Item, RunReport

__all__ = [
    "SkillEntry", "EntryMeta", "PlannerOutput", "SolverOutput", "CuratorOutput",
    "DualEncoder", "HashingEncoder", "ProjectionHead", "build_encoder",
    "CCQSTrainer", "CCQSPair", "SkillBook", "Retriever",
    "VerificationSignal", "build_signal", "Pipeline", "Item", "RunReport",
    "ToolConfig", "ToolResult", "ToolTranscript", "run_python",
]
