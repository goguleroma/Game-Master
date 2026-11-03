"""Memory consolidation pipeline: STM → MTM → LTM → GSM → GM.

This module implements the background consolidation process that promotes
memories across tiers based on significance, confidence, and confirmation.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .memory import (
    GridMemoryStore,
    KnowledgeItem,
    Scope,
    ShortTermMemory,
    SourceMemory,
)


@dataclass
class ConsolidationConfig:
    """Configuration thresholds for memory consolidation."""

    # STM → MTM thresholds
    stm_significance_threshold: float = 0.3
    stm_min_reward_magnitude: float = 0.1
    stm_novelty_threshold: float = 0.5
    stm_recurrence_threshold: int = 3

    # MTM → LTM thresholds
    mtm_confidence_threshold: float = 0.6
    mtm_confirmation_threshold: int = 2

    # LTM → GSM thresholds
    ltm_pattern_min_support: int = 5
    ltm_pattern_confidence: float = 0.7

    # GSM → GM thresholds
    gsm_generalization_min_games: int = 3
    gsm_generalization_confidence: float = 0.8
    gsm_generalization_min_evidence: int = 10

    # Consolidation timing
    consolidation_interval_seconds: float = 1.0
    max_consolidations_per_step: int = 10


@dataclass
class STMEvent:
    """Wrapper for STM frame with metadata for consolidation."""

    frame: Any
    timestamp: float
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    reward: float = 0.0
    action: Optional[str] = None
    observation: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "reward": self.reward,
            "action": self.action,
            "observation": self.observation,
            "metadata": self.metadata,
        }


class MemoryConsolidator:
    """Background consolidation pipeline: STM → MTM → LTM → GSM → GM.

    Runs as an async background task, evaluating and promoting memories
    across tiers based on configurable thresholds.
    """

    def __init__(
        self,
        memory_store: GridMemoryStore,
        config: Optional[ConsolidationConfig] = None,
        on_promotion: Optional[Callable[[str, str, KnowledgeItem], None]] = None,
    ) -> None:
        """Initialize the consolidator.

        Args:
            memory_store: The GridMemoryStore containing all memory tiers.
            config: Consolidation thresholds and timing configuration.
            on_promotion: Optional callback when promotion occurs (source_tier, target_tier, item).
        """
        self.memory = memory_store
        self.config = config or ConsolidationConfig()
        self.on_promotion = on_promotion

        # Internal tracking
        self._stm_events: List[STMEvent] = []
        self._mtm_hypotheses: Dict[str, KnowledgeItem] = {}  # key -> hypothesis
        self._ltm_experience_ids: Set[str] = set()
        self._gsm_patterns: Dict[str, List[KnowledgeItem]] = defaultdict(list)
        self._gm_patterns: Dict[str, KnowledgeItem] = {}

        # Background task
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

        # Statistics
        self.stats = {
            "stm_evaluated": 0,
            "stm_to_mtm": 0,
            "mtm_to_ltm": 0,
            "ltm_to_gsm": 0,
            "gsm_to_gm": 0,
            "consolidation_cycles": 0,
        }

    async def start(self) -> None:
        """Start the background consolidation loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._consolidation_loop())

    async def stop(self) -> None:
        """Stop the background consolidation loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _consolidation_loop(self) -> None:
        """Background loop running consolidation steps at intervals."""
        while self._running:
            try:
                await self.consolidate_step()
            except Exception as e:
                # Log but don't crash the loop
                print(f"[MemoryConsolidator] Error in consolidation step: {e}")
            await asyncio.sleep(self.config.consolidation_interval_seconds)

    def add_stm_event(
        self,
        frame: Any,
        reward: float = 0.0,
        action: Optional[str] = None,
        observation: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None,
    ) -> STMEvent:
        """Add an event to STM for consolidation evaluation."""
        event = STMEvent(
            frame=frame,
            timestamp=timestamp or time.monotonic(),
            reward=reward,
            action=action,
            observation=observation,
            metadata=metadata or {},
        )
        self._stm_events.append(event)
        # Also add to the actual STM store
        self.memory.short_term.add(frame, timestamp=event.timestamp)
        return event

    async def consolidate_step(self) -> Dict[str, int]:
        """Run one consolidation cycle across all tiers.

        Returns:
            Dictionary with counts of promotions at each tier.
        """
        async with self._lock:
            results = {
                "stm_to_mtm": 0,
                "mtm_to_ltm": 0,
                "ltm_to_gsm": 0,
                "gsm_to_gm": 0,
            }

            # STM → MTM: Evaluate and promote significant events
            results["stm_to_mtm"] = await self._consolidate_stm_to_mtm()

            # MTM → LTM: Consolidate confirmed hypotheses
            results["mtm_to_ltm"] = await self._consolidate_mtm_to_ltm()

            # LTM → GSM: Extract game-specific patterns
            results["ltm_to_gsm"] = await self._consolidate_ltm_to_gsm()

            # GSM → GM: Generalize cross-game patterns
            results["gsm_to_gm"] = await self._consolidate_gsm_to_gm()

            # Update stats
            for key, value in results.items():
                self.stats[key] += value
            self.stats["consolidation_cycles"] += 1

            return results

    # ============================================================
    # Significance Evaluation
    # ============================================================

    def evaluate_significance(self, event: STMEvent) -> float:
        """Evaluate the significance of an STM event for promotion.

        Considers:
        - Reward magnitude (absolute value)
        - Novelty (unseen state/action combinations)
        - Recurrence (repeated similar events)

        Returns:
            Significance score in [0, 1].
        """
        significance = 0.0

        # Reward magnitude contribution
        reward_mag = abs(event.reward)
        if reward_mag >= self.config.stm_min_reward_magnitude:
            significance += min(1.0, reward_mag) * 0.4

        # Novelty contribution
        novelty = self._compute_novelty(event)
        if novelty >= self.config.stm_novelty_threshold:
            significance += novelty * 0.3

        # Recurrence contribution
        recurrence = self._compute_recurrence(event)
        if recurrence >= self.config.stm_recurrence_threshold:
            significance += min(1.0, recurrence / 10.0) * 0.3

        return min(1.0, significance)

    def _compute_novelty(self, event: STMEvent) -> float:
        """Compute novelty score based on unseen state/action patterns."""
        if event.observation is None:
            return 0.0

        # Create a signature from observation + action
        obs_signature = self._observation_signature(event.observation)
        action = event.action or "none"
        signature = f"{action}:{obs_signature}"

        # Check against existing MTM hypotheses
        for hypothesis in self._mtm_hypotheses.values():
            if hypothesis.context.get("signature") == signature:
                return 0.0  # Already known

        # Check against LTM experiences
        for exp in self.memory.experiences:
            if exp.get("signature") == signature:
                return 0.0  # Already in LTM

        return 1.0  # Completely novel

    def _compute_recurrence(self, event: STMEvent) -> int:
        """Count how many similar events occurred recently."""
        if event.observation is None:
            return 0

        obs_signature = self._observation_signature(event.observation)
        action = event.action or "none"
        signature = f"{action}:{obs_signature}"

        count = 0
        for e in self._stm_events:
            if e.event_id == event.event_id:
                continue
            e_sig = self._observation_signature(e.observation) if e.observation else ""
            e_action = e.action or "none"
            if f"{e_action}:{e_sig}" == signature:
                count += 1

        return count

    def _observation_signature(self, observation: Dict[str, Any]) -> str:
        """Create a compact signature from observation."""
        # Extract key features for signature
        parts = []
        if "grid" in observation:
            grid = observation["grid"]
            if isinstance(grid, dict):
                # Simplified: hash of visible cell types
                cell_types = sorted(set(str(v) for v in grid.values()))
                parts.append("grid:" + ",".join(cell_types[:5]))
        if "hud" in observation:
            hud = observation["hud"]
            if isinstance(hud, dict):
                health = hud.get("health")
                if health is not None:
                    parts.append(f"hp:{health}")
        return "|".join(parts) if parts else "empty"

    # ============================================================
    # STM → MTM Promotion
    # ============================================================

    async def _consolidate_stm_to_mtm(self) -> int:
        """Evaluate STM events and promote significant ones to MTM as hypotheses."""
        promoted = 0
        remaining_events = []

        for event in self._stm_events:
            significance = self.evaluate_significance(event)

            if significance >= self.config.stm_significance_threshold:
                confidence = significance
                self.promote_stm_to_mtm(event, confidence)
                promoted += 1

                if promoted >= self.config.max_consolidations_per_step:
                    remaining_events.append(event)
                    break
            else:
                remaining_events.append(event)

        self._stm_events = remaining_events
        self.stats["stm_evaluated"] += len(self._stm_events) + promoted
        return promoted

    def promote_stm_to_mtm(self, event: STMEvent, confidence: float) -> KnowledgeItem:
        """Promote a significant STM event to MTM as a hypothesis.

        Args:
            event: The STM event to promote.
            confidence: Initial confidence in [0, 1].

        Returns:
            The created KnowledgeItem hypothesis in MTM.
        """
        # Create a unique key for this hypothesis
        sig = self._observation_signature(event.observation) if event.observation else "unknown"
        action = event.action or "none"
        hypothesis_key = f"hypothesis:{action}:{sig}:{event.event_id[:8]}"

        hypothesis = KnowledgeItem(
            key=hypothesis_key,
            value={
                "event": event.to_dict(),
                "predicted_outcome": "unknown",
                "signature": f"{action}:{sig}",
            },
            confidence=confidence,
            evidence_count=1,
            context={
                "signature": f"{action}:{sig}",
                "source_event_id": event.event_id,
                "reward": event.reward,
            },
            provenance={
                "source_tier": SourceMemory.STM.value,
                "source_event_id": event.event_id,
                "promoted_at": time.monotonic(),
            },
            scope=Scope.EPISODE,
            confirmation_count=0,
            consolidated_from=[event.event_id],
        )

        self._mtm_hypotheses[hypothesis_key] = hypothesis
        self.memory.knowledge.append(hypothesis)

        if self.on_promotion:
            self.on_promotion(SourceMemory.STM.value, SourceMemory.MTM.value, hypothesis)

        return hypothesis

    # ============================================================
    # MTM → LTM Promotion
    # ============================================================

    async def _consolidate_mtm_to_ltm(self) -> int:
        """Consolidate confirmed MTM hypotheses to LTM as experiences."""
        promoted = 0
        to_remove = []

        for key, hypothesis in self._mtm_hypotheses.items():
            if (
                hypothesis.confidence >= self.config.mtm_confidence_threshold
                and hypothesis.confirmation_count >= self.config.mtm_confirmation_threshold
            ):
                self.promote_mtm_to_ltm(hypothesis)
                promoted += 1
                to_remove.append(key)

                if promoted >= self.config.max_consolidations_per_step:
                    break

        for key in to_remove:
            del self._mtm_hypotheses[key]

        self.stats["mtm_to_ltm"] += promoted
        return promoted

    def promote_mtm_to_ltm(self, hypothesis: KnowledgeItem) -> Dict[str, Any]:
        """Promote a confirmed MTM hypothesis to LTM as an experience.

        Args:
            hypothesis: The MTM hypothesis to consolidate.

        Returns:
            The created experience record in LTM.
        """
        experience = {
            "experience_id": str(uuid.uuid4()),
            "hypothesis_key": hypothesis.key,
            "signature": hypothesis.context.get("signature", "unknown"),
            "action": hypothesis.context.get("action", "unknown"),
            "reward": hypothesis.context.get("reward", 0.0),
            "confidence": hypothesis.confidence,
            "evidence_count": hypothesis.evidence_count,
            "confirmation_count": hypothesis.confirmation_count,
            "context": hypothesis.context,
            "consolidated_from": hypothesis.consolidated_from.copy(),
            "consolidated_at": time.monotonic(),
            "scope": Scope.SESSION.value,
        }

        self.memory.experiences.append(experience)
        # Note: _ltm_experience_ids is updated in _consolidate_ltm_to_gsm after pattern extraction

        # Update hypothesis provenance
        hypothesis.provenance["promoted_to_ltm_at"] = time.monotonic()
        hypothesis.provenance["ltm_experience_id"] = experience["experience_id"]
        hypothesis.scope = Scope.SESSION

        if self.on_promotion:
            self.on_promotion(SourceMemory.MTM.value, SourceMemory.LTM.value, hypothesis)

        return experience

    def confirm_hypothesis(self, hypothesis_key: str, reward: float) -> bool:
        """Confirm or disconfirm an MTM hypothesis based on observed outcome.

        Args:
            hypothesis_key: Key of the hypothesis to confirm.
            reward: Observed reward (positive confirms, negative disconfirms).

        Returns:
            True if hypothesis was found and updated.
        """
        hypothesis = self._mtm_hypotheses.get(hypothesis_key)
        if hypothesis is None:
            # Check in main knowledge list
            for item in self.memory.knowledge:
                if item.key == hypothesis_key:
                    hypothesis = item
                    break

        if hypothesis is None:
            return False

        # Update confidence based on reward
        if reward > 0:
            hypothesis.confirmation_count += 1
            hypothesis.confidence = min(1.0, hypothesis.confidence + 0.1)
        else:
            hypothesis.confidence = max(0.0, hypothesis.confidence - 0.15)

        hypothesis.evidence_count += 1
        return True

    # ============================================================
    # LTM → GSM Pattern Extraction
    # ============================================================

    async def _consolidate_ltm_to_gsm(self) -> int:
        """Extract game-specific patterns from LTM experiences to GSM."""
        # Group experiences by signature/type
        pattern_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for exp in self.memory.experiences:
            if exp.get("experience_id") in self._ltm_experience_ids:
                continue  # Already processed
            sig = exp.get("signature", "unknown")
            pattern_groups[sig].append(exp)

        extracted = 0
        for signature, experiences in pattern_groups.items():
            if len(experiences) >= self.config.ltm_pattern_min_support:
                # Compute aggregate confidence
                avg_confidence = sum(e.get("confidence", 0) for e in experiences) / len(experiences)
                if avg_confidence >= self.config.ltm_pattern_confidence:
                    self.extract_gsm_patterns(experiences)
                    extracted += 1

                    # Mark as processed
                    for exp in experiences:
                        self._ltm_experience_ids.add(exp.get("experience_id", ""))

                    if extracted >= self.config.max_consolidations_per_step:
                        break

        self.stats["ltm_to_gsm"] += extracted
        return extracted

    def extract_gsm_patterns(self, ltm_experiences: List[Dict[str, Any]]) -> List[KnowledgeItem]:
        """Extract game-specific patterns (controls, mechanics, maps) from LTM experiences.

        Args:
            ltm_experiences: List of LTM experience records.

        Returns:
            List of created GSM KnowledgeItems.
        """
        if not ltm_experiences:
            return []

        # Determine pattern type from experiences
        first_exp = ltm_experiences[0]
        signature = first_exp.get("signature", "unknown")
        action = first_exp.get("action", "unknown")

        patterns = []

        # Control pattern: action -> effect mapping
        control_pattern = KnowledgeItem(
            key=f"gsm:control:{signature}",
            value={
                "type": "control_mapping",
                "action": action,
                "signature": signature,
                "sample_count": len(ltm_experiences),
                "avg_reward": sum(e.get("reward", 0) for e in ltm_experiences) / len(ltm_experiences),
                "reward_range": [
                    min(e.get("reward", 0) for e in ltm_experiences),
                    max(e.get("reward", 0) for e in ltm_experiences),
                ],
            },
            confidence=sum(e.get("confidence", 0) for e in ltm_experiences) / len(ltm_experiences),
            evidence_count=len(ltm_experiences),
            context={
                "pattern_type": "control",
                "game_id": self.memory.game_id,
                "source_experiences": [e.get("experience_id") for e in ltm_experiences],
            },
            provenance={
                "source_tier": SourceMemory.LTM.value,
                "extracted_at": time.monotonic(),
                "source_count": len(ltm_experiences),
            },
            scope=Scope.GAME,
            confirmation_count=len(ltm_experiences),
            consolidated_from=[e.get("experience_id", "") for e in ltm_experiences],
        )
        patterns.append(control_pattern)
        self._gsm_patterns["control"].append(control_pattern)
        self.memory.general_knowledge.append(control_pattern)

        # Mechanic pattern: state transition rules
        mechanic_pattern = KnowledgeItem(
            key=f"gsm:mechanic:{signature}",
            value={
                "type": "mechanic",
                "signature": signature,
                "conditions": first_exp.get("context", {}),
                "sample_count": len(ltm_experiences),
            },
            confidence=sum(e.get("confidence", 0) for e in ltm_experiences) / len(ltm_experiences),
            evidence_count=len(ltm_experiences),
            context={
                "pattern_type": "mechanic",
                "game_id": self.memory.game_id,
                "source_experiences": [e.get("experience_id") for e in ltm_experiences],
            },
            provenance={
                "source_tier": SourceMemory.LTM.value,
                "extracted_at": time.monotonic(),
                "source_count": len(ltm_experiences),
            },
            scope=Scope.GAME,
            confirmation_count=len(ltm_experiences),
            consolidated_from=[e.get("experience_id", "") for e in ltm_experiences],
        )
        patterns.append(mechanic_pattern)
        self._gsm_patterns["mechanic"].append(mechanic_pattern)
        self.memory.general_knowledge.append(mechanic_pattern)

        if self.on_promotion:
            for p in patterns:
                self.on_promotion(SourceMemory.LTM.value, SourceMemory.GSM.value, p)

        return patterns

    # ============================================================
    # GSM → GM Generalization
    # ============================================================

    async def _consolidate_gsm_to_gm(self) -> int:
        """Generalize cross-game patterns from GSM to GM."""
        generalized = 0

        # Group GSM patterns by type across games
        pattern_by_type: Dict[str, Dict[str, List[KnowledgeItem]]] = defaultdict(
            lambda: defaultdict(list)
        )

        for item in self.memory.general_knowledge:
            if item.scope != Scope.GAME:
                continue
            pattern_type = item.context.get("pattern_type", "unknown")
            game_id = item.context.get("game_id", "unknown")
            signature = item.value.get("signature", "unknown") if isinstance(item.value, dict) else "unknown"
            pattern_by_type[pattern_type][signature].append(item)

        # For each pattern type and signature, check if we have enough games
        for pattern_type, signatures in pattern_by_type.items():
            for signature, items in signatures.items():
                game_ids = set(item.context.get("game_id") for item in items)
                total_evidence = sum(item.evidence_count for item in items)
                avg_confidence = sum(item.confidence for item in items) / len(items)

                if (
                    len(game_ids) >= self.config.gsm_generalization_min_games
                    and avg_confidence >= self.config.gsm_generalization_confidence
                    and total_evidence >= self.config.gsm_generalization_min_evidence
                ):
                    self.generalize_to_gm(pattern_type, signature, items)
                    generalized += 1

                    if generalized >= self.config.max_consolidations_per_step:
                        return generalized

        self.stats["gsm_to_gm"] += generalized
        return generalized

    def generalize_to_gm(
        self,
        pattern_type: str,
        signature: str,
        gsm_items: List[KnowledgeItem],
    ) -> KnowledgeItem:
        """Generalize a cross-game pattern from GSM to GM.

        Args:
            pattern_type: Type of pattern (control, mechanic, map).
            signature: Pattern signature.
            gsm_items: List of GSM KnowledgeItems from different games.

        Returns:
            The created GM KnowledgeItem.
        """
        game_ids = list(set(item.context.get("game_id") for item in gsm_items))
        all_source_experiences = []
        for item in gsm_items:
            all_source_experiences.extend(item.consolidated_from)

        gm_pattern = KnowledgeItem(
            key=f"gm:{pattern_type}:{signature}",
            value={
                "type": pattern_type,
                "signature": signature,
                "games_observed": game_ids,
                "game_count": len(game_ids),
                "total_evidence": sum(item.evidence_count for item in gsm_items),
                "avg_confidence": sum(item.confidence for item in gsm_items) / len(gsm_items),
                "transferable": True,
            },
            confidence=sum(item.confidence for item in gsm_items) / len(gsm_items),
            evidence_count=sum(item.evidence_count for item in gsm_items),
            context={
                "pattern_type": pattern_type,
                "source_games": game_ids,
                "source_gsm_keys": [item.key for item in gsm_items],
            },
            provenance={
                "source_tier": SourceMemory.GSM.value,
                "generalized_at": time.monotonic(),
                "source_count": len(gsm_items),
            },
            scope=Scope.GENERAL,
            confirmation_count=sum(item.confirmation_count for item in gsm_items),
            transferable=True,
            consolidated_from=all_source_experiences,
        )

        self._gm_patterns[f"{pattern_type}:{signature}"] = gm_pattern
        self.memory.general_knowledge.append(gm_pattern)

        # Mark source GSM items as transferable
        for item in gsm_items:
            item.transferable = True
            item.provenance["generalized_to_gm_at"] = time.monotonic()
            item.provenance["gm_key"] = gm_pattern.key

        if self.on_promotion:
            self.on_promotion(SourceMemory.GSM.value, SourceMemory.GM.value, gm_pattern)

        return gm_pattern

    # ============================================================
    # Utility Methods
    # ============================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get consolidation statistics."""
        return {
            **self.stats,
            "stm_queue_size": len(self._stm_events),
            "mtm_hypotheses": len(self._mtm_hypotheses),
            "ltm_experiences": len(self._ltm_experience_ids),
            "gsm_patterns": {k: len(v) for k, v in self._gsm_patterns.items()},
            "gm_patterns": len(self._gm_patterns),
        }

    def get_traceability(self, item_key: str) -> Optional[List[str]]:
        """Get the full traceability chain for a consolidated item."""
        # Check all tiers
        for item in self.memory.knowledge:
            if item.key == item_key:
                return item.consolidated_from
        for item in self.memory.general_knowledge:
            if item.key == item_key:
                return item.consolidated_from
        for exp in self.memory.experiences:
            if exp.get("experience_id") == item_key or exp.get("hypothesis_key") == item_key:
                return exp.get("consolidated_from", [])
        return None

    async def flush_stm(self) -> int:
        """Force evaluation of all pending STM events."""
        count = 0
        while self._stm_events:
            result = await self._consolidate_stm_to_mtm()
            count += result
            if result == 0:
                break
        return count