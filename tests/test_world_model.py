"""Tests for World Model components (RSSM, imagination, reconstruction)."""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass


# Since the project doesn't have explicit RSSM implementation yet,
# we'll test the existing world model components and create tests
# that would apply to a future RSSM implementation.

from adapters.grid_30x30.world_model import (
    GridWorldModel,
    WorldState,
    WorldEntity,
    bfs_shortest_path,
)


class TestWorldState:
    """Tests for WorldState dataclass."""

    def test_default_state(self) -> None:
        state = WorldState()

        assert state.entities == []
        assert state.visible_cells == []
        assert state.unknown_cells == []
        assert state.frontier_cells == []
        assert state.current_position is None
        assert state.target_position is None
        assert state.obstacle_positions == []
        assert state.grid_size is None
        assert state.confidence == 0.0
        assert state.goal is None

    def test_state_with_values(self) -> None:
        state = WorldState(
            entities=[WorldEntity(kind="agent", position=(5, 5), confidence=1.0)],
            visible_cells=[(5, 5), (6, 5)],
            current_position=(5, 5),
            target_position=(10, 10),
            grid_size=(30, 30),
            confidence=0.9,
            goal="reach_target",
        )

        assert len(state.entities) == 1
        assert state.current_position == (5, 5)
        assert state.target_position == (10, 10)
        assert state.goal == "reach_target"


class TestWorldEntity:
    """Tests for WorldEntity dataclass."""

    def test_entity_creation(self) -> None:
        entity = WorldEntity(
            kind="enemy",
            position=(10, 10),
            confidence=0.8,
            metadata={"health": 8},
            entity_id="entity-2",
        )

        assert entity.kind == "enemy"
        assert entity.position == (10, 10)
        assert entity.confidence == 0.8
        assert entity.metadata == {"health": 8}
        assert entity.entity_id == "entity-2"

    def test_entity_defaults(self) -> None:
        entity = WorldEntity(kind="test")

        assert entity.kind == "test"
        assert entity.position is None
        assert entity.confidence == 0.0
        assert entity.metadata == {}
        assert entity.entity_id is None


class TestGridWorldModel:
    """Tests for GridWorldModel."""

    def test_update_from_full_observation(self) -> None:
        model = GridWorldModel()
        obs = {
            "hud": {
                "grid": {
                    "width": 30,
                    "height": 30,
                    "position": [15, 15],
                    "target": [25, 25],
                    "visible_cells": [[15, 15], [16, 15], [15, 16]],
                    "obstacles": [[17, 15]],
                    "entities": [
                        {"id": "entity-1", "position": [16, 15], "kind": "obstacle", "confidence": 1.0},
                        {"id": "entity-2", "position": [20, 20], "kind": "enemy", "health": 8, "confidence": 0.9},
                    ],
                }
            },
            "goal": "reach_target",
            "confidence": 1.0,
        }

        state = model.update(obs)

        assert state.current_position == (15, 15)
        assert state.target_position == (25, 25)
        assert state.grid_size == (30, 30)
        assert len(state.visible_cells) == 3
        assert state.obstacle_positions == [(17, 15)]
        assert len(state.entities) == 5  # agent, target, entity-1, entity-2, obstacle
        assert state.goal == "reach_target"
        assert state.confidence == 1.0

    def test_update_with_missing_fields(self) -> None:
        model = GridWorldModel()
        obs = {
            "hud": {
                "grid": {
                    "width": 30,
                    "height": 30,
                    "position": [5, 5],
                }
            },
            "confidence": 0.5,
        }

        state = model.update(obs)

        assert state.current_position == (5, 5)
        assert state.target_position is None
        assert state.grid_size == (30, 30)
        assert state.confidence == 0.5

    def test_update_with_none_observation(self) -> None:
        model = GridWorldModel()
        state = model.update(None, position=(10, 10))

        assert state.current_position == (10, 10)
        assert state.confidence == 0.0
        assert state.entities == []

    def test_entities_include_agent_and_target(self) -> None:
        model = GridWorldModel()
        obs = {
            "hud": {
                "grid": {
                    "width": 30,
                    "height": 30,
                    "position": [10, 10],
                    "target": [20, 20],
                    "visible_cells": [],
                    "obstacles": [],
                    "entities": [],
                }
            },
            "confidence": 1.0,
        }

        state = model.update(obs)

        # Should have agent entity
        agent_entities = [e for e in state.entities if e.kind == "agent"]
        assert len(agent_entities) == 1
        assert agent_entities[0].position == (10, 10)
        assert agent_entities[0].confidence == 1.0

        # Should have target entity
        target_entities = [e for e in state.entities if e.kind == "target"]
        assert len(target_entities) == 1
        assert target_entities[0].position == (20, 20)

    def test_obstacle_entities_created(self) -> None:
        model = GridWorldModel()
        obs = {
            "hud": {
                "grid": {
                    "width": 30,
                    "height": 30,
                    "position": [10, 10],
                    "visible_cells": [],
                    "obstacles": [[12, 10], [13, 10]],
                    "entities": [],
                }
            },
            "confidence": 1.0,
        }

        state = model.update(obs)

        obstacle_entities = [e for e in state.entities if e.kind == "obstacle"]
        assert len(obstacle_entities) == 2
        positions = {e.position for e in obstacle_entities}
        assert (12, 10) in positions
        assert (13, 10) in positions


class TestBFSShortestPath:
    """Tests for BFS pathfinding used in world model."""

    def test_direct_path(self) -> None:
        path = bfs_shortest_path((0, 0), (3, 0), set(), bounds=(10, 10))
        assert path == [(0, 0), (1, 0), (2, 0), (3, 0)]

    def test_path_around_obstacle(self) -> None:
        blocked = {(1, 0)}
        path = bfs_shortest_path((0, 0), (2, 0), blocked, bounds=(10, 10))
        # Should go around: (0,0) -> (0,1) -> (1,1) -> (2,1) -> (2,0) or similar
        assert path[0] == (0, 0)
        assert path[-1] == (2, 0)
        assert (1, 0) not in path

    def test_no_path_when_blocked(self) -> None:
        blocked = {(0, 1), (1, 0), (1, 1)}  # Surround start
        path = bfs_shortest_path((0, 0), (2, 2), blocked, bounds=(10, 10))
        assert path == []

    def test_path_respects_bounds(self) -> None:
        path = bfs_shortest_path((0, 0), (15, 15), set(), bounds=(10, 10))
        assert path == []

    def test_path_with_allowed_cells(self) -> None:
        allowed = {(0, 0), (1, 0), (2, 0)}
        path = bfs_shortest_path((0, 0), (2, 0), set(), bounds=(10, 10), allowed=allowed)
        assert path == [(0, 0), (1, 0), (2, 0)]

    def test_path_with_blocked_transitions(self) -> None:
        blocked_transitions = {((0, 0), (1, 0))}  # Can't go right from start
        path = bfs_shortest_path((0, 0), (2, 0), set(), bounds=(10, 10), blocked_transitions=blocked_transitions)
        # Should go up, right, right, down
        assert (1, 0) not in path
        assert path[-1] == (2, 0)

    def test_start_equals_goal(self) -> None:
        path = bfs_shortest_path((5, 5), (5, 5), set(), bounds=(10, 10))
        assert path == [(5, 5)]


# ============================================================
# RSSM-style World Model Tests (for future implementation)
# ============================================================

class MockRSSMWorldModel:
    """
    Mock implementation of RSSM (Recurrent State Space Model) for testing.
    This represents the interface a future RSSM implementation should have.
    """

    def __init__(self, state_dim: int = 256, action_dim: int = 6, hidden_dim: int = 512):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self._state = None

    def init_state(self, batch_size: int = 1) -> Dict[str, torch.Tensor]:
        """Initialize recurrent state."""
        return {
            "stochastic": torch.zeros(batch_size, self.state_dim),
            "deterministic": torch.zeros(batch_size, self.hidden_dim),
        }

    def observe_step(
        self,
        prev_state: Dict[str, torch.Tensor],
        action: torch.Tensor,
        observation: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Update state with observation (posterior)."""
        # Mock implementation
        return {
            "stochastic": torch.randn_like(prev_state["stochastic"]),
            "deterministic": torch.randn_like(prev_state["deterministic"]),
        }

    def imagine_step(
        self,
        prev_state: Dict[str, torch.Tensor],
        action: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Predict next state without observation (prior)."""
        return {
            "stochastic": torch.randn_like(prev_state["stochastic"]),
            "deterministic": torch.randn_like(prev_state["deterministic"]),
        }

    def get_features(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Get combined features for decoder/policy."""
        return torch.cat([state["stochastic"], state["deterministic"]], dim=-1)

    def decode_observation(self, features: torch.Tensor) -> torch.Tensor:
        """Reconstruct observation from features."""
        return torch.randn(features.shape[0], 3, 64, 64)

    def decode_reward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict reward from features."""
        return torch.randn(features.shape[0], 1)

    def decode_continue(self, features: torch.Tensor) -> torch.Tensor:
        """Predict continuation probability."""
        return torch.sigmoid(torch.randn(features.shape[0], 1))


class TestRSSMWorldModelInterface:
    """Tests for RSSM world model interface."""

    def test_init_state_shape(self) -> None:
        model = MockRSSMWorldModel(state_dim=32, hidden_dim=64)
        state = model.init_state(batch_size=4)

        assert state["stochastic"].shape == (4, 32)
        assert state["deterministic"].shape == (4, 64)

    def test_observe_step(self) -> None:
        model = MockRSSMWorldModel()
        state = model.init_state()
        action = torch.zeros(1, 6)
        obs = torch.randn(1, 3, 64, 64)

        new_state = model.observe_step(state, action, obs)

        assert "stochastic" in new_state
        assert "deterministic" in new_state
        assert new_state["stochastic"].shape == state["stochastic"].shape

    def test_imagine_step(self) -> None:
        model = MockRSSMWorldModel()
        state = model.init_state()
        action = torch.zeros(1, 6)

        new_state = model.imagine_step(state, action)

        assert "stochastic" in new_state
        assert "deterministic" in new_state

    def test_imagine_trajectory(self) -> None:
        model = MockRSSMWorldModel()
        state = model.init_state()
        actions = torch.randn(10, 1, 6)  # 10 steps

        states = []
        current = state
        for t in range(10):
            current = model.imagine_step(current, actions[t])
            states.append(current)

        assert len(states) == 10
        assert all("stochastic" in s for s in states)

    def test_decode_observation(self) -> None:
        model = MockRSSMWorldModel()
        features = torch.randn(2, 256)

        recon = model.decode_observation(features)

        assert recon.shape == (2, 3, 64, 64)

    def test_decode_reward(self) -> None:
        model = MockRSSMWorldModel()
        features = torch.randn(5, 256)

        reward = model.decode_reward(features)

        assert reward.shape == (5, 1)

    def test_decode_continue(self) -> None:
        model = MockRSSMWorldModel()
        features = torch.randn(3, 256)

        cont = model.decode_continue(features)

        assert cont.shape == (3, 1)
        assert (cont >= 0).all() and (cont <= 1).all()


# ============================================================
# Imagination/Planning Tests
# ============================================================

class TestImaginationBasedPlanning:
    """Tests for imagination-based planning using world model."""

    def test_plan_with_random_actions(self) -> None:
        model = MockRSSMWorldModel()
        state = model.init_state()
        horizon = 5

        # Random policy for planning
        def random_policy(features: torch.Tensor) -> torch.Tensor:
            return torch.randn(features.shape[0], 6)

        # Imagine trajectory
        states = [state]
        for _ in range(horizon):
            features = model.get_features(states[-1])
            action = random_policy(features)
            next_state = model.imagine_step(states[-1], action)
            states.append(next_state)

        assert len(states) == horizon + 1

    def test_plan_evaluates_cumulative_reward(self) -> None:
        model = MockRSSMWorldModel()
        state = model.init_state()
        horizon = 5

        def policy(features: torch.Tensor) -> torch.Tensor:
            return torch.zeros(features.shape[0], 6)  # Fixed action

        # Imagine and evaluate
        total_reward = 0.0
        current = state
        for _ in range(horizon):
            features = model.get_features(current)
            action = policy(features)
            current = model.imagine_step(current, action)
            reward = model.decode_reward(model.get_features(current))
            total_reward += reward.item()

        assert isinstance(total_reward, float)

    def test_multiple_trajectories_for_planning(self) -> None:
        model = MockRSSMWorldModel()
        num_trajectories = 10
        horizon = 5

        all_rewards = []
        for _ in range(num_trajectories):
            state = model.init_state()
            total = 0.0
            for _ in range(horizon):
                features = model.get_features(state)
                action = torch.randn(1, 6)
                state = model.imagine_step(state, action)
                reward = model.decode_reward(model.get_features(state))
                total += reward.item()
            all_rewards.append(total)

        assert len(all_rewards) == num_trajectories
        best_idx = np.argmax(all_rewards)
        assert 0 <= best_idx < num_trajectories


# ============================================================
# Reconstruction Loss Tests
# ============================================================

class TestReconstructionLoss:
    """Tests for observation reconstruction."""

    def test_reconstruction_loss_mse(self) -> None:
        model = MockRSSMWorldModel()
        batch_size = 4

        # Simulate encoding and decoding
        obs = torch.randn(batch_size, 3, 64, 64)
        features = torch.randn(batch_size, 256)
        recon = model.decode_observation(features)

        # MSE loss
        loss = torch.nn.functional.mse_loss(recon, obs)
        assert loss.item() >= 0

    def test_reconstruction_loss_l1(self) -> None:
        model = MockRSSMWorldModel()
        batch_size = 4

        obs = torch.randn(batch_size, 3, 64, 64)
        features = torch.randn(batch_size, 256)
        recon = model.decode_observation(features)

        loss = torch.nn.functional.l1_loss(recon, obs)
        assert loss.item() >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])