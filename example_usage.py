"""
Example: Complete Learning to Remember (LeRe) Agent

This script demonstrates how to initialize and run the full LeRe agent
with CCME, CRTE, online training, and memory operations.

Implements Algorithm 1 from the paper with the addition of the
forward-looking Synthesizer stage (DC-RS inspired).
"""

import numpy as np
from main import (
    # CCME
    MemoryEncoder,
    CCMERetriever,
    CCMELoss,
    # CRTE
    TrajectoryEncoder,
    ReflectionEncoder,
    CRTELoss,
    # Memory Operations
    MemoryOperations,
    # Training
    TrainingDataCollector,
    OnlineTrainer,
    # Agent
    LeReAgent
)
from main.utils import LeRePipeline, load_prompts_from_directory


def create_lere_agent(
    prompts_dir: str,
    llm_call_function,
    initial_memory: list = None,
    enable_training: bool = True,
    k_upd: int = 50
):
    """
    Create a fully configured LeRe Agent.

    Args:
        prompts_dir: Directory with generator/reflector/curator/synthesizer prompts
        llm_call_function: Function that takes prompt string and returns LLM response
        initial_memory: Starting memory bank
        enable_training: Enable online training
        k_upd: Update frequency

    Returns:
        LeReAgent instance
    """
    print("Initializing Learning to Remember (LeRe) Agent...")
    print("=" * 80)

    # 1. Initialize LLM Pipeline
    print("\n[1/8] Loading prompts...")
    llm_pipeline = load_prompts_from_directory(prompts_dir)
    print(f"  Loaded prompts from {prompts_dir}")

    # 2. Initialize CCME Components
    print("\n[2/8] Initializing CCME (Contrastive Contextual Memory Encoder)...")
    memory_encoder = MemoryEncoder(
        embedding_model="text-embedding-3-small",
        base_dim=1536,
        projection_dim=768,
        trainable=enable_training  # Trainable if online learning enabled
    )
    print(f"  MemoryEncoder initialized (trainable={enable_training})")

    ccme_retriever = CCMERetriever(
        memory_encoder=memory_encoder,
        alpha=0.7,  # Balance similarity (70%) vs reliability (30%)
        top_k=5     # Retrieve top 5 items
    )
    print(f"  CCMERetriever initialized (alpha=0.7, top_k=5)")

    ccme_loss = CCMELoss(
        temperature=0.07,
        adaptive_temperature=True
    )
    print(f"  CCMELoss initialized (temperature=0.07)")

    # 3. Initialize CRTE Components
    print("\n[3/8] Initializing CRTE (Contrastive Reflective Trajectory Encoder)...")
    trajectory_encoder = TrajectoryEncoder(
        embedding_model="text-embedding-3-small",
        base_dim=1536,
        projection_dim=768,
        d_temporal=64,
        trainable=enable_training
    )
    print(f"  TrajectoryEncoder initialized (trainable={enable_training})")

    reflection_encoder = ReflectionEncoder(
        embedding_model="text-embedding-3-small",
        base_dim=1536,
        projection_dim=768,
        trainable=enable_training
    )
    print(f"  ReflectionEncoder initialized (trainable={enable_training})")

    crte_loss = CRTELoss(
        temperature=0.07,
        lambda_cluster=0.1,
        lambda_margin=0.1,
        margin_eta=0.2
    )
    print(f"  CRTELoss initialized")

    # 4. Initialize Memory Operations
    print("\n[4/8] Initializing Memory Operations...")
    memory_operations = MemoryOperations(
        memory_encoder=memory_encoder,
        dedup_threshold=0.95,
        prune_reliability_threshold=0.3,
        temporal_decay_lambda=1e-4,
        shard_capacity=1000,
        min_usage_count=1
    )
    print(f"  MemoryOperations initialized")

    # 5. Initialize Training Data Collector
    print("\n[5/8] Initializing Training Data Collector...")
    data_collector = TrainingDataCollector(
        ccme_buffer_size=1000,
        crte_buffer_size=500,
        temporal_decay=0.9
    )
    print(f"  TrainingDataCollector initialized")

    # 6. Initialize Online Trainer
    print("\n[6/8] Initializing Online Trainer...")
    online_trainer = OnlineTrainer(
        memory_encoder=memory_encoder,
        trajectory_encoder=trajectory_encoder,
        reflection_encoder=reflection_encoder,
        ccme_loss=ccme_loss,
        crte_loss=crte_loss,
        data_collector=data_collector,
        k_upd=k_upd,
        num_train_steps=10,
        batch_size=32,
        learning_rate=1e-4,
        lambda_crte=1.0
    )
    print(f"  OnlineTrainer initialized (k_upd={k_upd})")

    # 7. Initialize Complete Agent
    print("\n[7/8] Assembling LeRe Agent...")
    agent = LeReAgent(
        llm_pipeline=llm_pipeline,
        llm_call_fn=llm_call_function,
        memory_encoder=memory_encoder,
        ccme_retriever=ccme_retriever,
        ccme_loss=ccme_loss,
        trajectory_encoder=trajectory_encoder,
        reflection_encoder=reflection_encoder,
        crte_loss=crte_loss,
        memory_operations=memory_operations,
        data_collector=data_collector,
        online_trainer=online_trainer,
        initial_memory_bank=initial_memory,
        k_upd=k_upd,
        enable_training=enable_training,
        verbose=True
    )
    print(f"  LeReAgent initialized")

    # 8. Summary
    print("\n[8/8] Initialization Complete!")
    print("=" * 80)
    print(f"\nAgent Configuration:")
    print(f"  - CCME: Memory encoding + retrieval")
    print(f"  - CRTE: Trajectory-reflection alignment")
    print(f"  - Synthesizer: Forward-looking memory briefing (DC-RS inspired)")
    print(f"  - Memory Operations: Deduplication, pruning, capacity control")
    print(f"  - Online Training: {'ENABLED' if enable_training else 'DISABLED'}")
    print(f"  - Update Frequency: Every {k_upd} queries")
    print(f"  - Initial Memory: {len(initial_memory) if initial_memory else 0} items")
    print(f"\n{'='*80}\n")

    return agent


def example_usage():
    """
    Example usage of LeRe Agent.
    """

    # Mock LLM call function (replace with actual LLM API)
    def mock_llm_call(prompt: str) -> str:
        """Mock LLM that returns dummy responses."""
        if "SYNTHESIZER" in prompt or "synthesis" in prompt.lower():
            return """
<synthesis>
{
  "next_question_analysis": {
    "detected_domains": ["algorithms", "recursion"],
    "detected_skills": ["divide_conquer", "tree_traversal"],
    "estimated_difficulty": "MEDIUM",
    "key_challenges": ["identifying base cases", "choosing traversal order"]
  },
  "selected_items": [
    {
      "item_id": "m_001",
      "title": "Divide and Conquer Strategy",
      "relevance_tier": "HIGH",
      "relevance_reason": "Directly applicable to recursive tree problems",
      "key_bullets": ["Break problem into smaller subproblems", "Solve subproblems recursively"],
      "application_hint": "Apply DFS with recursive decomposition",
      "reliability": 0.8,
      "cautions": null
    }
  ],
  "cross_item_strategies": [],
  "conflict_warnings": [],
  "briefing_summary": "Use divide and conquer with recursive tree traversal. Memory item m_001 is highly relevant."
}
</synthesis>
"""
        elif "GENERATOR" in prompt or "solve" in prompt.lower():
            return """
<trajectory>
<step id="1" type="analysis" memory_refs="[]" timestamp="T1">
Analyzing the problem structure and constraints.
</step>
<step id="2" type="strategy" memory_refs="[m_001]" timestamp="T2">
Applying the divide and conquer strategy from memory.
</step>
</trajectory>

<self_assessment>
The approach seems solid and should lead to a correct solution.
</self_assessment>

<execution_status>SUCCESS</execution_status>

MEMORY ITEMS CONSULTED: [m_001]

FINAL ANSWER:
The solution is 42.
"""
        elif "REFLECTOR" in prompt or "reflect" in prompt.lower():
            return """
<reflection>
{
  "execution_status": "SUCCESS",
  "memory_evaluation": [
    {
      "item_id": "m_001",
      "verdict": "HELPFUL",
      "usage_context": "Step 2",
      "confusable_with": [],
      "update_recommendation": {
        "action": "INCREMENT_HELPFUL",
        "reason": "Strategy led to successful solution"
      }
    }
  ],
  "trajectory_analysis": {
    "critical_steps": [
      {
        "step_id": "2",
        "temporal_position": "early",
        "impact": "POSITIVE",
        "memory_influence": ["m_001"],
        "observation": "Key strategic decision"
      }
    ]
  },
  "new_insights": [
    {
      "type": "strategy",
      "title": "Divide and conquer pattern",
      "description": "Works well for recursive problems",
      "applicability": "Problems with substructure",
      "estimated_generalizability": "high"
    }
  ]
}
</reflection>
"""
        else:  # Curator
            return """
<curation>
{
  "query_id": "Q_0001",
  "reliability_updates": [
    {
      "item_id": "m_001",
      "action": "INCREMENT_HELPFUL"
    }
  ],
  "new_items": [],
  "updates_to_existing": [],
  "summary": {
    "items_to_add": 0,
    "items_to_update": 0,
    "reliability_updates": 1
  }
}
</curation>
"""

    # Initial memory bank
    initial_memory = [
        {
            "id": "m_001",
            "title": "Divide and Conquer Strategy",
            "bullets": [
                "Break problem into smaller subproblems",
                "Solve subproblems recursively",
                "Combine solutions"
            ],
            "tags": ["algorithms", "recursion"],
            "meta": {
                "helpful": 3,
                "harmful": 0,
                "used": 3,
                "last_used": "2025-01-13",
                "source_queries": ["Q_0000"],
                "confidence": "high"
            }
        }
    ]

    # Create agent
    agent = create_lere_agent(
        prompts_dir="./prompts",
        llm_call_function=mock_llm_call,
        initial_memory=initial_memory,
        enable_training=True,  # Enable online training
        k_upd=3  # Update every 3 queries (for demo)
    )

    # Run some example queries
    queries = [
        "How do I solve a recursive tree traversal problem?",
        "What's the best approach for dynamic programming?",
        "How can I optimize a backtracking algorithm?",
        "Explain merge sort algorithm",
        "How to handle graph cycles?"
    ]

    print("\n" + "=" * 80)
    print("Running Example Queries")
    print("=" * 80)

    results = agent.run_batch(queries)

    # Print summary
    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)

    stats = agent.get_statistics()
    print(f"\nTotal Queries: {stats['total_queries']}")
    print(f"Success Rate: {stats['success_rate']:.2%}")
    print(f"Memory Size: {stats['memory_size']} items")
    print(f"Training Updates: {stats['training_updates']}")
    print(f"Buffer Stats: {stats['buffer_stats']}")

    # Save agent state
    agent.save_state("lere_agent_checkpoint.pt")

    print("\n Example completed successfully!\n")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Learning to Remember (LeRe) Agent - Complete Example")
    print("=" * 80 + "\n")

    example_usage()
