"""Tests for modern DQN implementation."""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock
from typing import List, Tuple

from learning.dqn import (
    _QNetwork,
    DQNAgent,
    _PrioritizedTransition,
    _Transition,
)


class TestQNetwork:
    """Tests for _QNetwork."""

    def test_init_default(self) -> None:
        net = _QNetwork(input_size=40, action_count=6)

        assert net.input_size == 40
        assert net.action_count == 6
        assert net.hidden_size == 512
        assert net.num_layers == 3
        assert net.use_spatial is False

    def test_init_custom(self) -> None:
        net = _QNetwork(
            input_size=40,
            action_count=6,
            hidden_size=256,
            num_layers=2,
            use_spatial=True,
            spatial_shape=(5, 8),
            num_aux_heads=2,
            dropout=0.2,
        )

        assert net.hidden_size == 256
        assert net.num_layers == 2
        assert net.use_spatial is True
        assert net.spatial_shape == (5, 8)
        assert net.num_aux_heads == 2
        assert net.dropout == 0.2

    def test_init_spatial_shape_mismatch_raises(self) -> None:
        with pytest.raises(AssertionError):
            _QNetwork(input_size=40, action_count=6, use_spatial=True, spatial_shape=(4, 8))  # 4*8=32 != 40

    def test_forward_returns_q_values(self) -> None:
        net = _QNetwork(input_size=40, action_count=6)
        state = torch.randn(2, 40)  # batch of 2

        q_values = net(state)

        assert q_values.shape == (2, 6)

    def test_get_q_values(self) -> None:
        net = _QNetwork(input_size=40, action_count=6)
        state = torch.randn(1, 40)

        q = net.get_q_values(state)

        assert q.shape == (1, 6)

    def test_get_policy_logits(self) -> None:
        net = _QNetwork(input_size=40, action_count=6)
        state = torch.randn(1, 40)

        logits = net.get_policy_logits(state)

        assert logits.shape == (1, 6)

    def test_get_value(self) -> None:
        net = _QNetwork(input_size=40, action_count=6)
        state = torch.randn(3, 40)

        value = net.get_value(state)

        assert value.shape == (3,)

    def test_get_advantage(self) -> None:
        net = _QNetwork(input_size=40, action_count=6)
        state = torch.randn(1, 40)

        advantage = net.get_advantage(state)

        assert advantage.shape == (1, 6)

    def test_get_aux_predictions(self) -> None:
        net = _QNetwork(input_size=40, action_count=6, num_aux_heads=3)
        state = torch.randn(1, 40)

        aux = net.get_aux_predictions(state)

        assert len(aux) == 3
        assert all(a.shape == (1,) for a in aux)

    def test_forward_all(self) -> None:
        net = _QNetwork(input_size=40, action_count=6, num_aux_heads=2)
        state = torch.randn(2, 40)

        outputs = net.forward_all(state)

        assert "q_values" in outputs
        assert "policy_logits" in outputs
        assert "value" in outputs
        assert "aux" in outputs
        assert "features" in outputs
        assert outputs["q_values"].shape == (2, 6)
        assert outputs["policy_logits"].shape == (2, 6)
        assert outputs["value"].shape == (2,)
        assert len(outputs["aux"]) == 2
        assert outputs["features"].shape[0] == 2

    def test_spatial_mode(self) -> None:
        net = _QNetwork(input_size=40, action_count=6, use_spatial=False)
        assert net.use_spatial is False

        net.enable_spatial_mode()
        assert net.use_spatial is True

        net.disable_spatial_mode()
        assert net.use_spatial is False

    def test_spatial_forward(self) -> None:
        net = _QNetwork(input_size=40, action_count=6, use_spatial=True, spatial_shape=(5, 8))
        state = torch.randn(2, 40)  # flat input

        q = net.get_q_values(state)

        assert q.shape == (2, 6)


class TestDQNAgent:
    """Tests for DQNAgent."""

    def test_init_validates_inputs(self) -> None:
        with pytest.raises(ValueError):
            DQNAgent(input_size=0, action_count=6)

        with pytest.raises(ValueError):
            DQNAgent(input_size=40, action_count=0)

        with pytest.raises(ValueError):
            DQNAgent(input_size=40, action_count=6, replay_capacity=-1)

        with pytest.raises(ValueError):
            DQNAgent(input_size=40, action_count=6, gamma=1.5)

        with pytest.raises(ValueError):
            DQNAgent(input_size=40, action_count=6, epsilon_end=1.5)

    def test_init_sets_defaults(self) -> None:
        agent = DQNAgent(input_size=40, action_count=6)

        assert agent.input_size == 40
        assert agent.action_count == 6
        assert agent.batch_size == 64
        assert agent.gamma == 0.98
        assert agent.epsilon_start == 1.0
        assert agent.epsilon_end == 0.05
        assert agent.warmup_steps == 128
        assert agent.target_update_steps == 250

    def test_epsilon_property(self) -> None:
        agent = DQNAgent(input_size=40, action_count=6, epsilon_decay_steps=1000)
        agent.environment_steps = 0
        assert agent.epsilon == pytest.approx(1.0)

        agent.environment_steps = 500
        assert agent.epsilon == pytest.approx(0.525)  # halfway

        agent.environment_steps = 1000
        assert agent.epsilon == pytest.approx(0.05)  # at end

        agent.environment_steps = 2000
        assert agent.epsilon == pytest.approx(0.05)  # clamped at end

    def test_select_action_explore(self) -> None:
        agent = DQNAgent(input_size=40, action_count=6, seed=42)
        state = tuple(np.random.rand(40).astype(np.float32))
        action_mask = (True, True, True, True, True, True)

        # Force exploration
        action = agent.select_action(state, action_mask, explore=True)

        assert 0 <= action < 6
        assert agent.environment_steps == 1

    def test_select_action_exploit(self) -> None:
        agent = DQNAgent(input_size=40, action_count=6, seed=42)
        state = tuple(np.random.rand(40).astype(np.float32))
        action_mask = (True, True, True, True, True, True)

        # Disable exploration
        action = agent.select_action(state, action_mask, explore=False)

        assert 0 <= action < 6
        assert agent.environment_steps == 0  # Not incremented when not exploring

    def test_select_action_respects_mask(self) -> None:
        agent = DQNAgent(input_size=40, action_count=6, seed=42)
        state = tuple(np.random.rand(40).astype(np.float32))
        # Only allow actions 2 and 4
        action_mask = (False, False, True, False, True, False)

        action = agent.select_action(state, action_mask, explore=False)

        assert action in [2, 4]

    def test_select_action_invalid_mask_raises(self) -> None:
        agent = DQNAgent(input_size=40, action_count=6)
        state = tuple(np.random.rand(40).astype(np.float32))

        # Wrong length
        with pytest.raises(ValueError):
            agent.select_action(state, (True, True, True))

        # All False
        with pytest.raises(ValueError):
            agent.select_action(state, (False, False, False, False, False, False))

    def test_remember_valid_transition(self) -> None:
        agent = DQNAgent(input_size=40, action_count=6, use_per=False)
        state = tuple(np.random.rand(40).astype(np.float32))
        next_state = tuple(np.random.rand(40).astype(np.float32))
        action_mask = (True,) * 6
        next_action_mask = (True,) * 6

        agent.remember(state, 3, 1.0, next_state, next_action_mask, False)

        assert len(agent.replay) == 1

    def test_remember_terminal_transition(self) -> None:
        agent = DQNAgent(input_size=40, action_count=6, use_per=False)
        state = tuple(np.random.rand(40).astype(np.float32))
        next_state = tuple(np.random.rand(40).astype(np.float32))
        action_mask = (True,) * 6
        next_action_mask = (False,) * 6  # No valid next actions (terminal)

        agent.remember(state, 3, 1.0, next_state, next_action_mask, True)

        assert len(agent.replay) == 1

    def test_remember_invalid_inputs_raise(self) -> None:
        agent = DQNAgent(input_size=40, action_count=6)
        state = tuple(np.random.rand(40).astype(np.float32))
        next_state = tuple(np.random.rand(40).astype(np.float32))
        action_mask = (True,) * 6
        next_action_mask = (True,) * 6

        # Invalid action
        with pytest.raises(ValueError):
            agent.remember(state, 10, 1.0, next_state, next_action_mask, False)

        # Non-finite reward
        with pytest.raises(ValueError):
            agent.remember(state, 3, float('inf'), next_state, next_action_mask, False)

        # Non-terminal but no valid next action
        with pytest.raises(ValueError):
            agent.remember(state, 3, 1.0, next_state, (False,) * 6, False)

    def test_train_step_returns_none_before_warmup(self) -> None:
        agent = DQNAgent(input_size=40, action_count=6, batch_size=32, warmup_steps=100)
        state = tuple(np.random.rand(40).astype(np.float32))
        action_mask = (True,) * 6

        # Add only a few transitions
        for _ in range(10):
            next_state = tuple(np.random.rand(40).astype(np.float32))
            agent.remember(state, 0, 0.0, next_state, action_mask, False)

        loss = agent.train_step()
        assert loss is None

    def test_train_step_returns_loss_after_warmup(self) -> None:
        agent = DQNAgent(input_size=40, action_count=6, batch_size=4, warmup_steps=8)
        action_mask = (True,) * 6

        # Add enough transitions
        for i in range(10):
            state = tuple(np.random.rand(40).astype(np.float32))
            next_state = tuple(np.random.rand(40).astype(np.float32))
            agent.remember(state, i % 6, float(i % 3), next_state, action_mask, i == 9)

        loss = agent.train_step()
        assert loss is not None
        assert isinstance(loss, float)
        assert agent.training_steps == 1

    def test_behavior_clone(self) -> None:
        agent = DQNAgent(input_size=40, action_count=6)
        state = tuple(np.random.rand(40).astype(np.float32))
        action_mask = (True,) * 6

        # Create demonstrations
        samples = [
            (state, 2, action_mask),
            (state, 4, action_mask),
            (state, 1, action_mask),
        ]

        loss = agent.behavior_clone(samples, epochs=2, batch_size=2)

        assert isinstance(loss, float)
        assert loss >= 0

    def test_behavior_clone_validates_inputs(self) -> None:
        agent = DQNAgent(input_size=40, action_count=6)
        state = tuple(np.random.rand(40).astype(np.float32))
        action_mask = (True,) * 6

        # Empty samples
        with pytest.raises(ValueError):
            agent.behavior_clone([])

        # Invalid action in sample
        with pytest.raises(ValueError):
            agent.behavior_clone([(state, 10, action_mask)])

        # Invalid mask in sample
        with pytest.raises(ValueError):
            agent.behavior_clone([(state, 2, (False,) * 6)])

    def test_save_load_roundtrip(self, tmp_path) -> None:
        agent = DQNAgent(input_size=40, action_count=6, seed=42)
        action_mask = (True,) * 6

        # Add some training data
        for i in range(20):
            state = tuple(np.random.rand(40).astype(np.float32))
            next_state = tuple(np.random.rand(40).astype(np.float32))
            agent.remember(state, i % 6, float(i % 3), next_state, action_mask, i == 19)

        # Train a bit
        for _ in range(5):
            agent.train_step()

        # Save
        save_path = tmp_path / "dqn_checkpoint.pt"
        agent.save(save_path, metadata={"test": "value"})

        # Load into new agent
        new_agent = DQNAgent(input_size=40, action_count=6, seed=123)
        metadata = new_agent.load(save_path, resume_training=True)

        assert metadata["test"] == "value"
        assert new_agent.environment_steps == agent.environment_steps
        assert new_agent.training_steps == agent.training_steps

        # Check weights match
        for p1, p2 in zip(agent.online.parameters(), new_agent.online.parameters()):
            assert torch.allclose(p1, p2)

    def test_save_load_architecture_mismatch(self, tmp_path) -> None:
        agent = DQNAgent(input_size=40, action_count=6, hidden_size=512)
        save_path = tmp_path / "dqn_checkpoint.pt"
        agent.save(save_path)

        # Try to load with different architecture
        new_agent = DQNAgent(input_size=40, action_count=6, hidden_size=256)

        with pytest.raises(ValueError, match="Architecture mismatch"):
            new_agent.load(save_path, resume_training=True)

    def test_select_action_validates_state(self) -> None:
        agent = DQNAgent(input_size=40, action_count=6)
        action_mask = (True,) * 6

        # Wrong size
        with pytest.raises(ValueError):
            agent.select_action(tuple([0.0] * 30), action_mask)

        # Non-finite values
        with pytest.raises(ValueError):
            agent.select_action(tuple([float('nan')] * 40), action_mask)


class TestDQNAgentPER:
    """Tests for DQNAgent with Prioritized Experience Replay."""

    def test_per_initialization(self) -> None:
        agent = DQNAgent(input_size=40, action_count=6, use_per=True)

        assert agent.use_per is True
        assert isinstance(agent.replay, list)
        assert agent._replay_priorities.shape == (20000,)  # default capacity

    def test_remember_with_per(self) -> None:
        agent = DQNAgent(input_size=40, action_count=6, use_per=True)
        state = tuple(np.random.rand(40).astype(np.float32))
        next_state = tuple(np.random.rand(40).astype(np.float32))
        action_mask = (True,) * 6

        agent.remember(state, 3, 1.0, next_state, action_mask, False)

        assert len(agent.replay) == 1
        assert isinstance(agent.replay[0], _PrioritizedTransition)
        assert agent._replay_filled == 1

    def test_train_step_updates_priorities(self) -> None:
        agent = DQNAgent(input_size=40, action_count=6, batch_size=4, warmup_steps=8, use_per=True)
        action_mask = (True,) * 6

        for i in range(10):
            state = tuple(np.random.rand(40).astype(np.float32))
            next_state = tuple(np.random.rand(40).astype(np.float32))
            agent.remember(state, i % 6, float(i % 3), next_state, action_mask, i == 9)

        initial_priorities = agent._replay_priorities[:agent._replay_filled].copy()
        loss = agent.train_step()

        assert loss is not None
        # Priorities should have been updated
        updated_priorities = agent._replay_priorities[:agent._replay_filled]
        assert not np.array_equal(initial_priorities, updated_priorities)

    def test_per_beta_annealing(self) -> None:
        agent = DQNAgent(input_size=40, action_count=6, batch_size=4, warmup_steps=8, use_per=True, per_beta_frames=100)
        action_mask = (True,) * 6

        for i in range(10):
            state = tuple(np.random.rand(40).astype(np.float32))
            next_state = tuple(np.random.rand(40).astype(np.float32))
            agent.remember(state, i % 6, float(i % 3), next_state, action_mask, False)

        initial_beta = agent.per_beta
        agent.train_step()
        assert agent.per_beta >= initial_beta  # Beta increases

        # After many steps, beta should approach 1.0
        for _ in range(200):
            agent.train_step()
        assert agent.per_beta == pytest.approx(1.0, abs=0.01)


class TestTransitionDataclasses:
    """Tests for transition dataclasses."""

    def test_prioritized_transition(self) -> None:
        trans = _PrioritizedTransition(
            state=tuple([1.0] * 40),
            action=3,
            reward=1.0,
            next_state=tuple([2.0] * 40),
            next_action_mask=(True,) * 6,
            done=False,
            priority=0.5,
            idx=0,
        )
        assert trans.state == tuple([1.0] * 40)
        assert trans.priority == 0.5
        assert trans.idx == 0

    def test_transition(self) -> None:
        trans = _Transition(
            state=tuple([1.0] * 40),
            action=3,
            reward=1.0,
            next_state=tuple([2.0] * 40),
            next_action_mask=(True,) * 6,
            done=False,
        )
        assert trans.state == tuple([1.0] * 40)
        assert trans.done is False


class TestDQNAgentDeviceHandling:
    """Tests for device handling in DQNAgent."""

    def test_device_cpu(self) -> None:
        agent = DQNAgent(input_size=40, action_count=6, device=torch.device("cpu"))
        assert agent.device == torch.device("cpu")

    @patch('learning.dqn.DIRECTML_AVAILABLE', False)
    @patch('learning.dqn.torch.cuda.is_available', False)
    def test_device_auto_fallback_cpu(self) -> None:
        agent = DQNAgent(input_size=40, action_count=6, device=None)
        assert agent.device == torch.device("cpu")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])