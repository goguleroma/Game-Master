"""Tests for MemoryConsolidator."""

import asyncio
import time
from collections import defaultdict
from unittest.mock import Mock

import pytest

from simulated_game.agent.memory import GridMemoryStore, Scope, SourceMemory, KnowledgeItem
from simulated_game.agent.memory_consolidator import (
    ConsolidationConfig,
    MemoryConsolidator,
    STMEvent,
)


@pytest.fixture
def memory_store():
    """Create a test memory store."""
    return GridMemoryStore(game_id="test_game")


@pytest.fixture
def config():
    """Create a test configuration with low thresholds."""
    return ConsolidationConfig(
        stm_significance_threshold=0.1,
        stm_min_reward_magnitude=0.05,
        stm_novelty_threshold=0.1,
        stm_recurrence_threshold=1,
        mtm_confidence_threshold=0.3,  # Lower to allow promotion with moderate confidence
        mtm_confirmation_threshold=1,
        ltm_pattern_min_support=2,
        ltm_pattern_confidence=0.3,
        gsm_generalization_min_games=1,
        gsm_generalization_confidence=0.5,
        gsm_generalization_min_evidence=2,
        consolidation_interval_seconds=0.01,
        max_consolidations_per_step=100,
    )


@pytest.fixture
def consolidator(memory_store, config):
    """Create a test consolidator."""
    return MemoryConsolidator(memory_store, config)


class TestSTMEvent:
    """Tests for STMEvent."""

    def test_create_event(self):
        event = STMEvent(
            frame={"test": "data"},
            timestamp=time.monotonic(),
            reward=0.5,
            action="move_up",
            observation={"grid": {(0, 0): "empty"}},
        )
        assert event.reward == 0.5
        assert event.action == "move_up"
        assert "event_id" in event.to_dict()


class TestSignificanceEvaluation:
    """Tests for significance evaluation."""

    @pytest.mark.asyncio
    async def test_high_reward_increases_significance(self, consolidator):
        event = STMEvent(
            frame={},
            timestamp=time.monotonic(),
            reward=0.8,
            action="move_up",
            observation={"grid": {(0, 0): "empty"}},
        )
        significance = consolidator.evaluate_significance(event)
        assert significance > 0.3

    @pytest.mark.asyncio
    async def test_zero_reward_low_significance(self, consolidator):
        event = STMEvent(
            frame={},
            timestamp=time.monotonic(),
            reward=0.0,
            action="move_up",
            observation={"grid": {(0, 0): "empty"}},
        )
        significance = consolidator.evaluate_significance(event)
        # Novelty contributes 0.3, so significance == 0.3 at minimum for novel events
        assert significance <= 0.3

    @pytest.mark.asyncio
    async def test_novel_event_high_novelty(self, consolidator):
        event = STMEvent(
            frame={},
            timestamp=time.monotonic(),
            reward=0.1,
            action="new_action",
            observation={"grid": {(0, 0): "novel_type"}},
        )
        significance = consolidator.evaluate_significance(event)
        # Novelty should contribute
        assert significance > 0.2


class TestSTMTomTM:
    """Tests for STM to MTM promotion."""

    @pytest.mark.asyncio
    async def test_promote_significant_event(self, consolidator):
        event = consolidator.add_stm_event(
            frame={"test": "data"},
            reward=0.5,
            action="move_up",
            observation={"grid": {(0, 0): "empty"}},
        )

        promoted = await consolidator._consolidate_stm_to_mtm()
        assert promoted == 1
        assert len(consolidator._mtm_hypotheses) == 1

    @pytest.mark.asyncio
    async def test_insignificant_event_not_promoted(self, consolidator):
        # First, add and promote a similar event to make this one non-novel
        consolidator.add_stm_event(
            frame={"test": "data"},
            reward=0.5,
            action="move_up",
            observation={"grid": {(0, 0): "empty"}},
        )
        await consolidator._consolidate_stm_to_mtm()

        # Now add a zero-reward event with same signature - should be non-novel
        event = consolidator.add_stm_event(
            frame={"test": "data"},
            reward=0.0,
            action="move_up",
            observation={"grid": {(0, 0): "empty"}},
        )

        promoted = await consolidator._consolidate_stm_to_mtm()
        assert promoted == 0
        # The first hypothesis should still be in MTM (not confirmed), no new one added
        assert len(consolidator._mtm_hypotheses) == 1

    @pytest.mark.asyncio
    async def test_hypothesis_has_traceability(self, consolidator):
        event = consolidator.add_stm_event(
            frame={"test": "data"},
            reward=0.5,
            action="move_up",
            observation={"grid": {(0, 0): "empty"}},
        )

        await consolidator._consolidate_stm_to_mtm()
        hypothesis = list(consolidator._mtm_hypotheses.values())[0]

        assert hypothesis.consolidated_from == [event.event_id]
        assert hypothesis.provenance["source_tier"] == SourceMemory.STM.value
        assert hypothesis.scope == Scope.EPISODE


class TestMTMtoLTM:
    """Tests for MTM to LTM promotion."""

    @pytest.mark.asyncio
    async def test_confirmed_hypothesis_promoted(self, consolidator):
        # Add and promote to MTM
        event = consolidator.add_stm_event(
            frame={}, reward=0.5, action="move_up", observation={"grid": {(0, 0): "empty"}}
        )
        await consolidator._consolidate_stm_to_mtm()

        # Confirm the hypothesis
        hypothesis = list(consolidator._mtm_hypotheses.values())[0]
        consolidator.confirm_hypothesis(hypothesis.key, reward=0.5)

        # Consolidate to LTM
        promoted = await consolidator._consolidate_mtm_to_ltm()
        assert promoted == 1
        assert len(consolidator.memory.experiences) == 1

        experience = consolidator.memory.experiences[0]
        assert experience["hypothesis_key"] == hypothesis.key
        assert experience["consolidated_from"] == [event.event_id]

    @pytest.mark.asyncio
    async def test_unconfirmed_hypothesis_not_promoted(self, consolidator):
        event = consolidator.add_stm_event(
            frame={}, reward=0.5, action="move_up", observation={"grid": {(0, 0): "empty"}}
        )
        await consolidator._consolidate_stm_to_mtm()

        # Don't confirm - confidence too low
        promoted = await consolidator._consolidate_mtm_to_ltm()
        assert promoted == 0


class TestLTMtoGSM:
    """Tests for LTM to GSM pattern extraction."""

    @pytest.mark.asyncio
    async def test_extract_control_pattern(self, consolidator):
        # Create multiple similar experiences with higher reward for higher confidence
        for i in range(3):
            event = consolidator.add_stm_event(
                frame={},
                reward=0.7,  # Higher reward -> higher significance -> higher confidence
                action="move_up",
                observation={"grid": {(0, 0): "empty"}},
            )
            await consolidator._consolidate_stm_to_mtm()

            hypothesis = list(consolidator._mtm_hypotheses.values())[0]
            consolidator.confirm_hypothesis(hypothesis.key, reward=0.7)
            await consolidator._consolidate_mtm_to_ltm()

        # Extract GSM patterns
        extracted = await consolidator._consolidate_ltm_to_gsm()
        assert extracted > 0

        # Check GSM patterns created
        gsm_items = [k for k in consolidator.memory.general_knowledge if k.scope == Scope.GAME]
        assert len(gsm_items) > 0

        # Check traceability
        for item in gsm_items:
            assert len(item.consolidated_from) > 0
            assert item.provenance["source_tier"] == SourceMemory.LTM.value


class TestGSMtoGM:
    """Tests for GSM to GM generalization."""

    @pytest.mark.asyncio
    async def test_generalize_across_games(self, memory_store, config):
        from simulated_game.agent.memory import KnowledgeItem

        consolidator = MemoryConsolidator(memory_store, config)

        # Manually add GSM patterns from 3 different games (proper KnowledgeItems only)
        for game_id in ["game1", "game2", "game3"]:
            item = KnowledgeItem(
                key=f"gsm:control:test_sig_{game_id}",
                value={"type": "control_mapping", "signature": "test_sig"},
                confidence=0.8,
                evidence_count=5,
                context={"pattern_type": "control", "game_id": game_id},
                provenance={"source_tier": SourceMemory.LTM.value},
                scope=Scope.GAME,
                confirmation_count=5,
                consolidated_from=[f"exp_{game_id}_{i}" for i in range(5)],
            )
            consolidator.memory.general_knowledge.append(item)

        # Now generalize
        generalized = await consolidator._consolidate_gsm_to_gm()
        assert generalized == 1

        # Check GM pattern
        gm_items = [k for k in consolidator.memory.general_knowledge if k.scope == Scope.GENERAL]
        assert len(gm_items) == 1
        gm = gm_items[0]
        assert gm.transferable is True
        assert gm.scope == Scope.GENERAL
        assert len(gm.consolidated_from) == 15  # 3 games * 5 experiences


class TestFullPipeline:
    """Integration tests for the full consolidation pipeline."""

    @pytest.mark.asyncio
    async def test_full_consolidation_cycle(self, consolidator):
        # Add multiple events
        for i in range(5):
            consolidator.add_stm_event(
                frame={},
                reward=0.4,
                action="move_up",
                observation={"grid": {(0, 0): "empty"}},
            )

        # Run multiple consolidation steps
        for _ in range(5):
            await consolidator.consolidate_step()

        stats = consolidator.get_stats()
        assert stats["stm_to_mtm"] > 0
        # MTM -> LTM needs confirmations
        # LTM -> GSM needs multiple experiences

    @pytest.mark.asyncio
    async def test_background_loop(self, consolidator):
        await consolidator.start()
        assert consolidator._running

        # Add event
        consolidator.add_stm_event(frame={}, reward=0.5, action="test")
        await asyncio.sleep(0.05)  # Let loop run

        await consolidator.stop()
        assert not consolidator._running


class TestTraceability:
    """Tests for traceability across tiers."""

    @pytest.mark.asyncio
    async def test_full_trace_chain(self, consolidator):
        # STM event - use higher reward for reliable consolidation
        event = consolidator.add_stm_event(
            frame={}, reward=0.7, action="move_up", observation={"grid": {(0, 0): "empty"}}
        )
        await consolidator._consolidate_stm_to_mtm()

        hypothesis = list(consolidator._mtm_hypotheses.values())[0]
        consolidator.confirm_hypothesis(hypothesis.key, reward=0.7)
        await consolidator._consolidate_mtm_to_ltm()

        experience = consolidator.memory.experiences[0]
        exp_id = experience["experience_id"]

        # Create more experiences for GSM - same high reward
        for _ in range(2):
            e = consolidator.add_stm_event(
                frame={}, reward=0.7, action="move_up", observation={"grid": {(0, 0): "empty"}}
            )
            await consolidator._consolidate_stm_to_mtm()
            h = list(consolidator._mtm_hypotheses.values())[0]
            consolidator.confirm_hypothesis(h.key, reward=0.7)
            await consolidator._consolidate_mtm_to_ltm()

        await consolidator._consolidate_ltm_to_gsm()

        # Check traceability from GSM back to LTM experiences
        gsm_items = [k for k in consolidator.memory.general_knowledge if k.scope == Scope.GAME]
        for item in gsm_items:
            trace = consolidator.get_traceability(item.key)
            assert trace is not None
            # Trace should contain LTM experience IDs (3 experiences)
            assert len(trace) == 3
            # Each experience ID should be traceable back to STM
            for exp_id in trace:
                exp_trace = consolidator.get_traceability(exp_id)
                assert exp_trace is not None
                assert len(exp_trace) == 1  # Each experience from one STM event


class TestConfiguration:
    """Tests for configuration thresholds."""

    @pytest.mark.asyncio
    async def test_custom_thresholds(self, memory_store):
        config = ConsolidationConfig(
            stm_significance_threshold=0.9,  # Very high
            mtm_confidence_threshold=0.9,
            mtm_confirmation_threshold=5,
        )
        consolidator = MemoryConsolidator(memory_store, config)

        event = consolidator.add_stm_event(
            frame={}, reward=0.5, action="test", observation={}
        )
        promoted = await consolidator._consolidate_stm_to_mtm()
        assert promoted == 0  # Threshold too high


class TestKnowledgeDecay:
    """Tests for knowledge decay and superseding."""

    def test_add_knowledge_with_scope_and_transferable(self, memory_store):
        """Test adding knowledge with scope and transferable flag."""
        item = memory_store.add_knowledge(
            key="test_fact",
            value={"fact": "move_up goes up"},
            confidence=0.8,
            scope=Scope.GAME,
            transferable=True,
        )
        assert item.scope == Scope.GAME
        assert item.transferable is True
        assert item.validity["timestamp"] > 0
        assert item.validity["decay_rate"] == 0.01
        assert item.validity["confirmation_count"] == 0

    def test_confirm_knowledge(self, memory_store):
        """Test confirming knowledge increases confidence."""
        memory_store.add_knowledge(
            key="test_fact",
            value={"fact": "move_up goes up"},
            confidence=0.5,
        )
        result = memory_store.confirm_knowledge("test_fact", confidence_delta=0.2)
        assert result is True

        # Find the item
        item = next(k for k in memory_store.knowledge if k.key == "test_fact")
        assert item.confidence == 0.7
        assert item.validity["confirmation_count"] == 1

    def test_confirm_nonexistent_knowledge(self, memory_store):
        """Test confirming non-existent knowledge returns False."""
        result = memory_store.confirm_knowledge("nonexistent")
        assert result is False

    def test_supersede_knowledge(self, memory_store):
        """Test superseding knowledge maintains traceability."""
        # Add initial knowledge
        memory_store.add_knowledge(
            key="old_fact",
            value={"fact": "move_up goes up"},
            confidence=0.6,
        )

        # Supersede with new knowledge
        new_item = memory_store.supersede_knowledge(
            old_key="old_fact",
            new_key="new_fact",
            new_value={"fact": "move_up goes up unless blocked"},
            new_confidence=0.8,
        )

        # Check new item
        assert new_item.key == "new_fact"
        assert new_item.confidence == 0.8
        assert new_item.provenance.get("supersedes") == "old_fact"

        # Check old item marked as superseded
        old_item = next(k for k in memory_store.knowledge if k.key == "old_fact")
        assert old_item.validity["superseded_by"] == "new_fact"

    def test_get_valid_knowledge_filters_superseded(self, memory_store):
        """Test get_valid_knowledge filters out superseded items."""
        memory_store.add_knowledge(key="fact1", value="v1", confidence=0.8)
        memory_store.add_knowledge(key="fact2", value="v2", confidence=0.7)
        memory_store.supersede_knowledge("fact1", "fact3", "v3", 0.9)

        valid = memory_store.get_valid_knowledge()
        valid_keys = {k.key for k in valid}
        assert "fact1" not in valid_keys  # Superseded
        assert "fact2" in valid_keys
        assert "fact3" in valid_keys

    def test_get_valid_knowledge_filters_by_scope(self, memory_store):
        """Test get_valid_knowledge filters by scope."""
        memory_store.add_knowledge(key="ep_fact", value="v1", confidence=0.8, scope=Scope.EPISODE)
        memory_store.add_knowledge(key="game_fact", value="v2", confidence=0.7, scope=Scope.GAME)
        memory_store.add_knowledge(key="gen_fact", value="v3", confidence=0.6, scope=Scope.GENERAL)

        episode_valid = memory_store.get_valid_knowledge(scope=Scope.EPISODE)
        game_valid = memory_store.get_valid_knowledge(scope=Scope.GAME)
        general_valid = memory_store.get_valid_knowledge(scope=Scope.GENERAL)

        assert len(episode_valid) == 1
        assert episode_valid[0].key == "ep_fact"
        assert len(game_valid) == 1
        assert game_valid[0].key == "game_fact"
        assert len(general_valid) == 1
        assert general_valid[0].key == "gen_fact"

    def test_prune_invalid_knowledge(self, memory_store):
        """Test pruning removes superseded and low-confidence items."""
        memory_store.add_knowledge(key="fact1", value="v1", confidence=0.8)
        memory_store.add_knowledge(key="fact2", value="v2", confidence=0.05)  # Low confidence
        memory_store.add_knowledge(key="fact3", value="v3", confidence=0.7)
        memory_store.supersede_knowledge("fact3", "fact4", "v4", 0.9)

        removed = memory_store.prune_invalid_knowledge()
        assert removed == 2  # fact2 (low conf) and fact3 (superseded)

        remaining_keys = {k.key for k in memory_store.knowledge}
        assert "fact1" in remaining_keys
        assert "fact2" not in remaining_keys
        assert "fact3" not in remaining_keys
        assert "fact4" in remaining_keys


class TestGMPersistence:
    """Tests for GM (General Memory) persistence to SQLite."""

    def test_save_load_general_knowledge(self, tmp_path):
        """Test saving and loading GENERAL scope knowledge."""
        db_path = tmp_path / "test_gm.db"
        memory = GridMemoryStore(game_id="test", database_path=db_path)

        # Add some GENERAL knowledge
        memory.general_knowledge.append(
            KnowledgeItem(
                key="gm_fact1",
                value={"principle": "movement is relative"},
                confidence=0.9,
                evidence_count=5,
                context={"source_games": ["grid_v1", "grid_v2"]},
                provenance={"source_tier": "GM", "added_at": time.monotonic()},
                scope=Scope.GENERAL,
                transferable=True,
            )
        )
        memory.general_knowledge.append(
            KnowledgeItem(
                key="gm_fact2",
                value={"principle": "keys open doors"},
                confidence=0.8,
                evidence_count=3,
                context={"source_games": ["maze"]},
                provenance={"source_tier": "GM", "added_at": time.monotonic()},
                scope=Scope.GENERAL,
                transferable=True,
            )
        )

        # Save
        saved = memory.save_general_knowledge()
        assert saved == 2

        # Create new memory store and load
        memory2 = GridMemoryStore(game_id="test2", database_path=db_path)
        loaded = memory2.load_general_knowledge()
        assert loaded == 2

        # Verify loaded items
        keys = {k.key for k in memory2.general_knowledge}
        assert "gm_fact1" in keys
        assert "gm_fact2" in keys

        item1 = next(k for k in memory2.general_knowledge if k.key == "gm_fact1")
        assert item1.confidence == 0.9
        assert item1.scope == Scope.GENERAL
        assert item1.transferable is True
        assert "movement is relative" in item1.value.get("principle", "")

    def test_save_load_gm_patterns(self, tmp_path):
        """Test saving and loading GM patterns."""
        db_path = tmp_path / "test_gm.db"
        memory = GridMemoryStore(game_id="test", database_path=db_path)

        gm_patterns = {
            "control": [
                {"key": "move_up", "value": {"direction": "up", "effect": "y-1"}, "confidence": 0.95,
                 "source_games": ["grid_v1", "grid_v2"], "evidence_count": 10},
                {"key": "move_down", "value": {"direction": "down", "effect": "y+1"}, "confidence": 0.9,
                 "source_games": ["grid_v1", "grid_v2"], "evidence_count": 8},
            ],
            "mechanic": [
                {"key": "key_door", "value": {"key": "opens", "door": "unlocks"}, "confidence": 0.85,
                 "source_games": ["maze"], "evidence_count": 5},
            ],
            "principle": [
                {"key": "exploration", "value": {"strategy": "visit_frontier"}, "confidence": 0.8,
                 "source_games": ["grid_v1", "maze", "platformer"], "evidence_count": 15},
            ],
        }

        saved = memory.save_gm_patterns(gm_patterns)
        assert saved == 4

        # Load into new memory store
        memory2 = GridMemoryStore(game_id="test2", database_path=db_path)
        loaded = memory2.load_gm_patterns()

        assert "control" in loaded
        assert "mechanic" in loaded
        assert "principle" in loaded
        assert len(loaded["control"]) == 2
        assert len(loaded["mechanic"]) == 1
        assert len(loaded["principle"]) == 1

        # Check ordering by confidence
        assert loaded["control"][0]["confidence"] >= loaded["control"][1]["confidence"]

        # Check specific pattern
        move_up = next(p for p in loaded["control"] if p["key"] == "move_up")
        assert move_up["confidence"] == 0.95
        assert move_up["source_games"] == ["grid_v1", "grid_v2"]

    def test_get_gm_persistence_stats(self, tmp_path):
        """Test getting GM persistence statistics."""
        db_path = tmp_path / "test_gm.db"
        memory = GridMemoryStore(game_id="test", database_path=db_path)

        # Empty stats
        stats = memory.get_gm_persistence_stats()
        assert stats["general_knowledge"] == 0
        assert stats["gm_patterns"] == 0

        # Add and save
        memory.general_knowledge.append(
            KnowledgeItem(
                key="gm1", value={}, confidence=0.8, evidence_count=1, context={},
                provenance={}, scope=Scope.GENERAL, transferable=True
            )
        )
        memory.save_general_knowledge()
        memory.save_gm_patterns({"control": [{"key": "c1", "value": {}, "confidence": 0.9,
                                               "source_games": ["g1"], "evidence_count": 1}]})

        stats = memory.get_gm_persistence_stats()
        assert stats["general_knowledge"] == 1
        assert stats["gm_patterns"] == 1


class TestTopDownAttention:
    """Tests for top-down attention API."""

    def test_request_observation(self, memory_store):
        """Test requesting observation of a region."""
        request = memory_store.request_observation(
            roi=((0, 0), (10, 10)),
            priority=0.8,
            reason="target spotted",
            source="planner",
        )

        assert request.roi == ((0, 0), (10, 10))
        assert request.priority == 0.8
        assert request.reason == "target spotted"
        assert request.source == "planner"

    def test_request_observation_defaults(self, memory_store):
        """Test request_observation with default parameters."""
        request = memory_store.request_observation(
            roi=((5, 5), (15, 15)),
        )

        assert request.priority == 0.5
        assert request.reason == ""
        assert request.source == "planner"

    def test_request_observation_priority_clamping(self, memory_store):
        """Test that priority is clamped to 0.0-1.0."""
        req_low = memory_store.request_observation(roi=((0,0),(5,5)), priority=-0.5)
        req_high = memory_store.request_observation(roi=((0,0),(5,5)), priority=1.5)

        assert req_low.priority == 0.0
        assert req_high.priority == 1.0

    def test_get_attention_requests(self, memory_store):
        """Test getting top attention requests."""
        memory_store.request_observation(roi=((0,0),(5,5)), priority=0.3, reason="low")
        memory_store.request_observation(roi=((10,10),(15,15)), priority=0.9, reason="high")
        memory_store.request_observation(roi=((20,20),(25,25)), priority=0.6, reason="medium")

        top = memory_store.get_attention_requests(n=2)
        assert len(top) == 2
        assert top[0].priority == 0.9
        assert top[1].priority == 0.6

    def test_clear_attention_requests(self, memory_store):
        """Test clearing attention requests."""
        memory_store.request_observation(roi=((0,0),(5,5)), priority=0.5)
        memory_store.request_observation(roi=((10,10),(15,15)), priority=0.8)

        assert len(memory_store.attention.requests) == 2

        memory_store.clear_attention_requests()

        assert len(memory_store.attention.requests) == 0

    def test_get_attention_for_position(self, memory_store):
        """Test getting attention requests for a specific position."""
        memory_store.request_observation(roi=((0,0),(10,10)), priority=0.5, reason="area1")
        memory_store.request_observation(roi=((20,20),(30,30)), priority=0.8, reason="area2")
        memory_store.request_observation(roi=((5,5),(15,15)), priority=0.9, reason="area3")

        # Position (5,5) should match area1 and area3
        matching = memory_store.get_attention_for_position((5, 5))
        assert len(matching) == 2
        # Should be sorted by priority
        assert matching[0].reason == "area3"  # priority 0.9
        assert matching[1].reason == "area1"  # priority 0.5

        # Position (25,25) should match area2
        matching = memory_store.get_attention_for_position((25, 25))
        assert len(matching) == 1
        assert matching[0].reason == "area2"

    def test_attention_stats(self, memory_store):
        """Test getting attention statistics."""
        memory_store.request_observation(roi=((0,0),(5,5)), priority=0.5)
        memory_store.request_observation(roi=((10,10),(15,15)), priority=0.8)

        stats = memory_store.get_attention_stats()
        assert stats["active_requests"] == 2
        assert stats["total_history"] == 2
        assert len(stats["top_priorities"]) == 2
        assert stats["top_priorities"][0] == 0.8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])