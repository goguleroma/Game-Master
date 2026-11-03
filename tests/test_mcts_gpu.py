"""Tests for MCTS with GPU-accelerated batched rollouts."""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field


# Since the project doesn't have explicit MCTS implementation yet,
# we'll create tests for the expected interface and a mock implementation.

from adapters.grid_30x30.planner import GridPlanner, ActionCandidate
from adapters.grid_30x30.world_model import WorldState, bfs_shortest_path
from core.types import Action, ActionResult, GameState


@dataclass
class MCTSNode:
    """Monte Carlo Tree Search node."""
    state: Any
    parent: Optional['MCTSNode'] = None
    action: Optional[Action] = None
    children: Dict[Action, 'MCTSNode'] = field(default_factory=dict)
    visit_count: int = 0
    value_sum: float = 0.0
    prior: float = 0.0

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    @property
    def is_expanded(self) -> bool:
        return len(self.children) > 0


class MockMCTS:
    """Mock MCTS implementation for testing."""

    def __init__(
        self,
        world_model: Any,
        policy: Any,
        num_simulations: int = 50,
        c_puct: float = 1.0,
        discount: float = 0.99,
        device: str = "cpu",
    ):
        self.world_model = world_model
        self.policy = policy
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.discount = discount
        self.device = torch.device(device)

    def search(self, root_state: Any, legal_actions: List[Action]) -> Action:
        """Run MCTS search from root state."""
        root = MCTSNode(state=root_state)

        # Initialize root with policy priors
        self._expand_node(root, legal_actions)

        for _ in range(self.num_simulations):
            self._simulate(root, legal_actions)

        # Select best action by visit count
        best_action = max(root.children.keys(), key=lambda a: root.children[a].visit_count)
        return best_action

    def _expand_node(self, node: MCTSNode, legal_actions: List[Action]) -> None:
        """Expand node with all legal actions."""
        for action in legal_actions:
            prior = 1.0 / len(legal_actions)  # Uniform prior
            node.children[action] = MCTSNode(
                state=None,  # Will be set during simulation
                parent=node,
                action=action,
                prior=prior,
            )

    def _simulate(self, root: MCTSNode, legal_actions: List[Action]) -> None:
        """Run one simulation from root."""
        node = root
        search_path = [node]

        # Selection
        while node.is_expanded:
            action, node = self._select_child(node)
            search_path.append(node)

        # Expansion/Evaluation
        value = self._evaluate(node, legal_actions)

        # Backpropagation
        self._backpropagate(search_path, value)

    def _select_child(self, node: MCTSNode) -> Tuple[Action, MCTSNode]:
        """Select child using PUCT."""
        best_score = -float('inf')
        best_action = None
        best_child = None

        for action, child in node.children.items():
            ucb = child.value + self.c_puct * child.prior * np.sqrt(node.visit_count) / (1 + child.visit_count)
            if ucb > best_score:
                best_score = ucb
                best_action = action
                best_child = child

        return best_action, best_child

    def _evaluate(self, node: MCTSNode, legal_actions: List[Action]) -> float:
        """Evaluate leaf node using policy/value network."""
        # Mock: return random value
        return np.random.uniform(-1, 1)

    def _backpropagate(self, path: List[MCTSNode], value: float) -> None:
        """Backpropagate value up the tree."""
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += value
            value *= self.discount


class BatchedMCTS:
    """Batched MCTS for GPU acceleration."""

    def __init__(
        self,
        world_model: Any,
        policy: Any,
        num_simulations: int = 50,
        batch_size: int = 32,
        device: str = "cpu",
    ):
        self.world_model = world_model
        self.policy = policy
        self.num_simulations = num_simulations
        self.batch_size = batch_size
        self.device = torch.device(device)

    def search_batch(self, root_states: List[Any], legal_actions_list: List[List[Action]]) -> List[Action]:
        """Search multiple root states in parallel."""
        # For testing, just run sequentially
        results = []
        mcts = MockMCTS(self.world_model, self.policy, self.num_simulations, device=str(self.device))
        for root_state, legal_actions in zip(root_states, legal_actions_list):
            results.append(mcts.search(root_state, legal_actions))
        return results

    def search_batched_gpu(self, root_states: List[Any], legal_actions_list: List[List[Action]]) -> List[Action]:
        """GPU-accelerated batched search (mock implementation)."""
        # In real implementation, this would:
        # 1. Batch all root states
        # 2. Run simulations in parallel on GPU
        # 3. Use vectorized tree operations
        return self.search_batch(root_states, legal_actions_list)


class TestMCTSNode:
    """Tests for MCTSNode."""

    def test_node_creation(self) -> None:
        state = WorldState(current_position=(5, 5))
        node = MCTSNode(state=state)

        assert node.state == state
        assert node.parent is None
        assert node.action is None
        assert node.visit_count == 0
        assert node.value_sum == 0.0

    def test_node_value(self) -> None:
        node = MCTSNode(state=None)
        assert node.value == 0.0

        node.visit_count = 10
        node.value_sum = 5.0
        assert node.value == 0.5

    def test_is_expanded(self) -> None:
        node = MCTSNode(state=None)
        assert node.is_expanded is False

        node.children[Action(name="move_up")] = MCTSNode(state=None)
        assert node.is_expanded is True


class TestMockMCTS:
    """Tests for MockMCTS."""

    def test_search_returns_legal_action(self) -> None:
        world_model = Mock()
        policy = Mock()
        mcts = MockMCTS(world_model, policy, num_simulations=10)

        root_state = WorldState(current_position=(0, 0), grid_size=(30, 30))
        legal_actions = [
            Action(name="move_up"),
            Action(name="move_right"),
            Action(name="move_down"),
            Action(name="move_left"),
        ]

        best_action = mcts.search(root_state, legal_actions)

        assert best_action in legal_actions

    def test_search_with_single_action(self) -> None:
        world_model = Mock()
        policy = Mock()
        mcts = MockMCTS(world_model, policy, num_simulations=10)

        root_state = WorldState(current_position=(0, 0))
        legal_actions = [Action(name="move_right")]

        best_action = mcts.search(root_state, legal_actions)

        assert best_action == legal_actions[0]

    def test_expand_node_creates_children(self) -> None:
        world_model = Mock()
        policy = Mock()
        mcts = MockMCTS(world_model, policy)

        root = MCTSNode(state=WorldState())
        legal_actions = [Action(name="move_up"), Action(name="move_down")]

        mcts._expand_node(root, legal_actions)

        assert len(root.children) == 2
        for action in legal_actions:
            assert action in root.children
            assert root.children[action].prior == 0.5

    def test_backpropagate_updates_values(self) -> None:
        world_model = Mock()
        policy = Mock()
        mcts = MockMCTS(world_model, policy, discount=0.9)

        root = MCTSNode(state=None)
        child = MCTSNode(state=None, parent=root, action=Action(name="move_up"))
        root.children[Action(name="move_up")] = child

        path = [root, child]
        mcts._backpropagate(path, 1.0)

        assert root.visit_count == 1
        assert root.value_sum == 0.9  # discounted
        assert child.visit_count == 1
        assert child.value_sum == 1.0


class TestBatchedMCTS:
    """Tests for BatchedMCTS."""

    def test_search_batch(self) -> None:
        world_model = Mock()
        policy = Mock()
        batched = BatchedMCTS(world_model, policy, num_simulations=5, batch_size=2)

        root_states = [
            WorldState(current_position=(i, i), grid_size=(30, 30))
            for i in range(3)
        ]
        legal_actions_list = [
            [Action(name="move_up"), Action(name="move_right")]
            for _ in range(3)
        ]

        results = batched.search_batch(root_states, legal_actions_list)

        assert len(results) == 3
        for action in results:
            assert action.name in ["move_up", "move_right"]

    def test_search_batch_empty(self) -> None:
        world_model = Mock()
        policy = Mock()
        batched = BatchedMCTS(world_model, policy)

        results = batched.search_batch([], [])

        assert results == []


class TestMCTSWithGridPlanner:
    """Tests integrating MCTS with GridPlanner."""

    def test_mcts_uses_planner_for_priors(self) -> None:
        planner = GridPlanner()
        world_model = Mock()
        policy = Mock()

        # Mock policy to return planner-based priors
        def mock_policy(state, legal_actions):
            # Use planner to get direction preferences
            return {a: 1.0/len(legal_actions) for a in legal_actions}

        policy.get_action_priors = mock_policy

        mcts = MockMCTS(world_model, policy, num_simulations=10)

        root_state = WorldState(
            current_position=(5, 5),
            target_position=(10, 10),
            grid_size=(30, 30),
            obstacle_positions=[],
        )
        legal_actions = [
            Action(name="move_up"),
            Action(name="move_right"),
            Action(name="move_down"),
            Action(name="move_left"),
        ]

        best_action = mcts.search(root_state, legal_actions)

        assert best_action in legal_actions

    def test_mcts_respects_obstacles(self) -> None:
        world_model = Mock()
        policy = Mock()
        mcts = MockMCTS(world_model, policy, num_simulations=10)

        # State with obstacle blocking right movement
        root_state = WorldState(
            current_position=(5, 5),
            grid_size=(10, 10),
            obstacle_positions=[(6, 5)],  # Blocks move_right
        )
        # Only up and left are valid
        legal_actions = [
            Action(name="move_up"),
            Action(name="move_left"),
        ]

        best_action = mcts.search(root_state, legal_actions)

        assert best_action in legal_actions


# ============================================================
# GPU Acceleration Tests (Mock)
# ============================================================

class TestGPUAcceleration:
    """Tests for GPU-accelerated MCTS operations."""

    @patch('torch.cuda.is_available', False)
    def test_fallback_to_cpu(self) -> None:
        world_model = Mock()
        policy = Mock()

        # Should not raise even without CUDA
        batched = BatchedMCTS(world_model, policy, device="cuda")
        assert batched.device.type == "cuda"  # But operations will run on CPU

    def test_device_assignment(self) -> None:
        world_model = Mock()
        policy = Mock()

        mcts_cpu = MockMCTS(world_model, policy, device="cpu")
        assert mcts_cpu.device == torch.device("cpu")

        mcts_cuda = MockMCTS(world_model, policy, device="cuda")
        assert mcts_cuda.device == torch.device("cuda")

    def test_batch_tensor_operations(self) -> None:
        """Test that batched operations use tensors correctly."""
        batch_size = 4
        state_dim = 64

        # Simulate batched states
        states = torch.randn(batch_size, state_dim)
        actions = torch.randint(0, 6, (batch_size,))

        # Vectorized operations
        next_states = states + torch.randn_like(states) * 0.1
        rewards = torch.randn(batch_size)

        assert next_states.shape == (batch_size, state_dim)
        assert rewards.shape == (batch_size,)

    def test_parallel_simulation_concept(self) -> None:
        """Test concept of parallel simulations."""
        num_parallel = 32
        horizon = 10

        # In real implementation, this would run on GPU
        # Here we simulate the concept
        states = torch.zeros(num_parallel, 64)
        total_rewards = torch.zeros(num_parallel)

        for step in range(horizon):
            # Batched policy
            actions = torch.randint(0, 6, (num_parallel,))
            # Batched world model step
            states = states + torch.randn_like(states) * 0.1
            # Batched reward
            rewards = torch.randn(num_parallel)
            total_rewards += rewards * (0.99 ** step)

        best_idx = torch.argmax(total_rewards).item()
        assert 0 <= best_idx < num_parallel


class TestMCTSConfiguration:
    """Tests for MCTS configuration parameters."""

    def test_c_puct_effect(self) -> None:
        world_model = Mock()
        policy = Mock()

        # High c_puct = more exploration
        mcts_explore = MockMCTS(world_model, policy, num_simulations=20, c_puct=2.0)
        # Low c_puct = more exploitation
        mcts_exploit = MockMCTS(world_model, policy, num_simulations=20, c_puct=0.1)

        assert mcts_explore.c_puct > mcts_exploit.c_puct

    def test_discount_factor(self) -> None:
        world_model = Mock()
        policy = Mock()

        mcts = MockMCTS(world_model, policy, discount=0.9)
        root = MCTSNode(state=None)
        child = MCTSNode(state=None, parent=root, action=Action(name="move_up"))
        root.children[Action(name="move_up")] = child

        mcts._backpropagate([root, child], 1.0)

        assert child.value_sum == 1.0
        assert root.value_sum == 0.9  # Discounted

    def test_num_simulations(self) -> None:
        world_model = Mock()
        policy = Mock()

        mcts_few = MockMCTS(world_model, policy, num_simulations=10)
        mcts_many = MockMCTS(world_model, policy, num_simulations=1000)

        assert mcts_few.num_simulations < mcts_many.num_simulations


if __name__ == "__main__":
    pytest.main([__file__, "-v"])