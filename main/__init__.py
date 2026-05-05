"""
Main package for the LeRe (Learn-to-Remember) framework.

Exports:
- CCME encoders (Eq query encoder, Em memory encoder) + loss
- CRTE encoders (trajectory and reflection encoders) + loss
- Memory operations with CCME intelligence
- Training data collection and buffers
- Online trainer for test-time adaptation
"""

from .ccme_encoder import QueryEncoder, MemoryEncoder, CCMERetriever, CCMELoss
from .crte_encoder import (
    TrajectoryEncoder,
    ReflectionEncoder,
    ConstrainedTemporalEmbedding,
    CRTELoss
)
from .memory_operations import (
    MemoryOperations,
    apply_memory_updates_with_ccme,
    refine_memory_with_clustering
)
from .training_data import (
    TrainingBuffer,
    CCMEPairExtractor,
    CRTEPairExtractor,
    TrainingDataCollector
)
from .online_trainer import OnlineTrainer

__all__ = [
    # CCME (paper lines 241-244, 251-252)
    'QueryEncoder',  # Eq: query encoder
    'MemoryEncoder',  # Em: memory encoder
    'CCMERetriever',
    'CCMELoss',

    # CRTE
    'TrajectoryEncoder',
    'ReflectionEncoder',
    'ConstrainedTemporalEmbedding',
    'CRTELoss',

    # Memory Operations
    'MemoryOperations',
    'apply_memory_updates_with_ccme',
    'refine_memory_with_clustering',

    # Training Data
    'TrainingBuffer',
    'CCMEPairExtractor',
    'CRTEPairExtractor',
    'TrainingDataCollector',

    # Online Training
    'OnlineTrainer',
]
