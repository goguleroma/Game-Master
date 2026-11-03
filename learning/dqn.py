from collections import deque
from dataclasses import dataclass
import math
from pathlib import Path
import random
from typing import Any, Optional

import numpy as np
import torch
from torch import nn

# Optional DirectML import
try:
    import torch_directml
    DIRECTML_AVAILABLE = True
except ImportError:
    DIRECTML_AVAILABLE = False


@dataclass
class _PrioritizedTransition:
    """Transition with priority for PER."""
    state: tuple[float, ...]
    action: int
    reward: float
    next_state: tuple[float, ...]
    next_action_mask: tuple[bool, ...]
    done: bool
    priority: float = 1.0
    idx: int = 0


@dataclass(frozen=True)
class _Transition:
    state: tuple[float, ...]
    action: int
    reward: float
    next_state: tuple[float, ...]
    next_action_mask: tuple[bool, ...]
    done: bool


class _QNetwork(nn.Module):
    """
    Expanded Q-Network with:
    - Larger backbone for GPU utilization (512→512→512)
    - Spatial structure support (Conv1d ready for CNN migration)
    - Parallel heads: Q-values, Policy, Auxiliary (reward/next-state/uncertainty)
    - CNN-ready: accepts both flat (40,) and spatial (C, H, W) inputs
    """
    def __init__(
        self,
        input_size: int,
        action_count: int,
        hidden_size: int = 512,
        num_layers: int = 3,
        use_spatial: bool = False,
        spatial_shape: tuple[int, int] | None = None,  # (H, W) for 40 features
        num_aux_heads: int = 3,  # reward_pred, next_state_pred, uncertainty
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.action_count = action_count
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_spatial = use_spatial
        self.spatial_shape = spatial_shape or (5, 8)  # 5x8 = 40
        self.num_aux_heads = num_aux_heads
        self.dropout = dropout
        
        # Calculate spatial dimensions (only if spatial mode is enabled)
        if use_spatial:
            c, h, w = 1, *self.spatial_shape
            assert c * h * w == input_size, f"spatial_shape {self.spatial_shape} must match input_size {input_size}"
        
        # --- Backbone: Shared feature extractor ---
        if use_spatial:
            # CNN-ready: Conv1d over spatial dimensions (becomes Conv2d for images)
            self.backbone = nn.Sequential(
                # Input: (B, 1, 40) -> treat as (B, 1, H, W) conceptually
                nn.Conv1d(1, 32, kernel_size=3, padding=1),  # (B, 32, 40)
                nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=3, padding=1),  # (B, 64, 40)
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(32),  # (B, 64, 32)
                nn.Flatten(),  # (B, 64*32=2048)
                nn.Linear(64 * 32, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            backbone_out = hidden_size
        else:
            # MLP backbone (current path, but wider)
            layers = [
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            for _ in range(num_layers - 1):
                layers.extend([
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ])
            self.backbone = nn.Sequential(*layers)
            backbone_out = hidden_size
        
        # --- Parallel Heads ---
        
        # 1. Q-Value Head (main DQN output)
        self.q_head = nn.Sequential(
            nn.Linear(backbone_out, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_count),
        )
        
        # 2. Policy Head (action probabilities - for distillation/auxiliary)
        self.policy_head = nn.Sequential(
            nn.Linear(backbone_out, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_count),
        )
        
        # 3. Auxiliary Heads (multi-task learning for richer representations)
        #    - Reward prediction
        #    - Next-state prediction (representation)
        #    - Uncertainty estimation
        self.aux_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(backbone_out, hidden_size // 4),
                nn.ReLU(),
                nn.Linear(hidden_size // 4, 1),  # scalar per aux task
            )
            for _ in range(num_aux_heads)
        ])
        
        # 4. State-value head (V(s) for advantage decomposition)
        self.value_head = nn.Sequential(
            nn.Linear(backbone_out, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Standard forward for DQN compatibility - returns Q-values only.
        """
        return self.get_q_values(state)
    
    def get_q_values(self, state: torch.Tensor) -> torch.Tensor:
        """Get Q-values (main DQN output)."""
        features = self._extract_features(state)
        return self.q_head(features)
    
    def get_policy_logits(self, state: torch.Tensor) -> torch.Tensor:
        """Get policy logits (for distillation/behavior cloning)."""
        features = self._extract_features(state)
        return self.policy_head(features)
    
    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """Get state value V(s)."""
        features = self._extract_features(state)
        return self.value_head(features).squeeze(-1)
    
    def get_advantage(self, state: torch.Tensor) -> torch.Tensor:
        """Get advantage A(s,a) = Q(s,a) - V(s)."""
        q = self.get_q_values(state)
        v = self.get_value(state).unsqueeze(-1)
        return q - v
    
    def get_aux_predictions(self, state: torch.Tensor) -> list[torch.Tensor]:
        """Get auxiliary task predictions."""
        features = self._extract_features(state)
        return [head(features).squeeze(-1) for head in self.aux_heads]
    
    def forward_all(self, state: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass returning all heads (for multi-task training)."""
        features = self._extract_features(state)
        return {
            "q_values": self.q_head(features),
            "policy_logits": self.policy_head(features),
            "value": self.value_head(features).squeeze(-1),
            "aux": [head(features).squeeze(-1) for head in self.aux_heads],
            "features": features,  # for representation learning
        }
    
    def _extract_features(self, state: torch.Tensor) -> torch.Tensor:
        """Shared feature extraction."""
        if self.use_spatial and state.dim() == 2:
            # Reshape flat input to spatial: (B, 40) -> (B, 1, 40)
            state = state.unsqueeze(1)
        return self.backbone(state)
    
    def enable_spatial_mode(self) -> None:
        """Switch to spatial/Conv1d mode (call before training on spatial data)."""
        self.use_spatial = True
    
    def disable_spatial_mode(self) -> None:
        """Switch back to flat MLP mode."""
        self.use_spatial = False


class DQNAgent:
    """Small Double-DQN learner; environment-specific features/rewards stay outside."""

    FORMAT_VERSION = 2

    def __init__(
        self,
        input_size: int,
        action_count: int,
        seed: int = 0,
        replay_capacity: int = 20_000,
        batch_size: int = 64,
        gamma: float = 0.98,
        learning_rate: float = 0.0005,
        epsilon_decay_steps: int = 20_000,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        warmup_steps: int = 128,
        target_update_steps: int = 250,
        device: Optional[torch.device] = None,
        # New architecture parameters
        hidden_size: int = 512,
        num_layers: int = 3,
        use_spatial: bool = False,
        spatial_shape: tuple[int, int] | None = None,
        num_aux_heads: int = 3,
        dropout: float = 0.1,
        # Multi-task loss weights
        aux_loss_weight: float = 0.1,
        policy_distill_weight: float = 0.05,
        # Prioritized Experience Replay
        use_per: bool = True,
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        per_beta_frames: int = 100_000,
        per_eps: float = 1e-6,
    ) -> None:
        if input_size <= 0 or action_count <= 0:
            raise ValueError("input_size and action_count must be positive.")
        if replay_capacity <= 0 or batch_size <= 0:
            raise ValueError("Replay capacity and batch size must be positive.")
        if not 0 <= gamma <= 1:
            raise ValueError("gamma must be between zero and one.")
        if min(epsilon_decay_steps, warmup_steps, target_update_steps) <= 0:
            raise ValueError("Training step intervals must be positive.")
        if not 0 <= epsilon_end <= epsilon_start <= 1:
            raise ValueError("Epsilon values must satisfy 0 <= end <= start <= 1.")
        if not 0 <= per_alpha <= 1:
            raise ValueError("per_alpha must be between 0 and 1.")
        if not 0 <= per_beta_start <= 1:
            raise ValueError("per_beta_start must be between 0 and 1.")
        if per_beta_frames <= 0:
            raise ValueError("per_beta_frames must be positive.")
        self.input_size = input_size
        self.action_count = action_count
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.warmup_steps = warmup_steps
        self.target_update_steps = target_update_steps
        # Multi-task weights
        self.aux_loss_weight = aux_loss_weight
        self.policy_distill_weight = policy_distill_weight
        # PER parameters
        self.use_per = use_per
        self.per_alpha = per_alpha
        self.per_beta = per_beta_start
        self.per_beta_start = per_beta_start
        self.per_beta_frames = per_beta_frames
        self.per_eps = per_eps
        self._rng = random.Random(seed)
        # Use a local torch Generator for weight initialization and sampling
        # to avoid polluting the global PyTorch RNG state
        self._torch_generator = torch.Generator()
        self._torch_generator.manual_seed(seed)
        
        # Determine device
        if device is not None:
            self.device = device
        elif DIRECTML_AVAILABLE and torch_directml.is_available():
            self.device = torch_directml.device()
        else:
            self.device = torch.device("cpu")
        
        # Create expanded networks
        self.online = _QNetwork(
            input_size, action_count,
            hidden_size=hidden_size,
            num_layers=num_layers,
            use_spatial=use_spatial,
            spatial_shape=spatial_shape,
            num_aux_heads=num_aux_heads,
            dropout=dropout,
        ).to(self.device)
        self.target = _QNetwork(
            input_size, action_count,
            hidden_size=hidden_size,
            num_layers=num_layers,
            use_spatial=use_spatial,
            spatial_shape=spatial_shape,
            num_aux_heads=num_aux_heads,
            dropout=dropout,
        ).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        
        # OPTIMIZER ON CPU PATTERN: Keep optimizer state on CPU to avoid DirectML lerp fallback
        # Forward/backward runs on GPU, optimizer step runs on CPU
        if str(self.device).startswith("privateuseone"):  # DirectML device
            # Create optimizer with parameters on CPU
            cpu_params = [p.detach().cpu().requires_grad_(True) for p in self.online.parameters()]
            self.optimizer = torch.optim.Adam(cpu_params, lr=learning_rate)
            self._optimizer_on_cpu = True
            self._gpu_params = list(self.online.parameters())  # Reference to GPU params
        else:
            self.optimizer = torch.optim.Adam(self.online.parameters(), lr=learning_rate)
            self._optimizer_on_cpu = False
            self._gpu_params = None
        
        # Replay buffer - use list for PER (need random access for priority updates)
        self.replay_capacity = replay_capacity
        if self.use_per:
            self.replay: list[_PrioritizedTransition] = []
            self._replay_priorities = np.zeros(replay_capacity, dtype=np.float32)
            self._replay_pos = 0
            self._replay_filled = 0
        else:
            self.replay: deque[_Transition] = deque(maxlen=replay_capacity)
        self.environment_steps = 0
        self.training_steps = 0
        self._per_frame = 0  # For beta annealing

    @property
    def epsilon(self) -> float:
        progress = min(self.environment_steps / self.epsilon_decay_steps, 1.0)
        return self.epsilon_start - (self.epsilon_start - self.epsilon_end) * progress

    def select_action(
        self,
        state: tuple[float, ...],
        action_mask: tuple[bool, ...],
        explore: bool = True,
    ) -> int:
        self._validate_state(state)
        if len(action_mask) != self.action_count or not any(action_mask):
            raise ValueError("Action mask must match action_count and allow an action.")
        valid_actions = [
            index for index, valid in enumerate(action_mask) if valid
        ]
        if explore and self._rng.random() < self.epsilon:
            action = self._rng.choice(valid_actions)
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
                mask_tensor = torch.tensor(action_mask, dtype=torch.bool, device=self.device)
                values = self.online(state_tensor)
                values = values.masked_fill(~mask_tensor, -torch.inf)
                action = int(torch.argmax(values).item())
        if explore:
            self.environment_steps += 1
        return action

    def remember(
        self,
        state: tuple[float, ...],
        action: int,
        reward: float,
        next_state: tuple[float, ...],
        next_action_mask: tuple[bool, ...],
        done: bool,
    ) -> None:
        self._validate_state(state)
        self._validate_state(next_state)
        if not 0 <= action < self.action_count:
            raise ValueError("Action index is outside the configured action space.")
        if len(next_action_mask) != self.action_count:
            raise ValueError("Next action mask must match action_count.")
        if not math.isfinite(reward):
            raise ValueError("Reward must be finite.")
        if not done and not any(next_action_mask):
            raise ValueError("Non-terminal transitions require a valid next action.")
        
        if self.use_per:
            # Max priority for new transitions (will be updated after first training)
            max_priority = self._replay_priorities[:self._replay_filled].max() if self._replay_filled > 0 else 1.0
            transition = _PrioritizedTransition(
                state=state,
                action=action,
                reward=float(reward),
                next_state=next_state,
                next_action_mask=next_action_mask,
                done=done,
                priority=max_priority,
                idx=self._replay_pos,
            )
            if self._replay_filled < self.replay_capacity:
                self.replay.append(transition)
                self._replay_filled += 1
            else:
                self.replay[self._replay_pos] = transition
            self._replay_priorities[self._replay_pos] = max_priority
            self._replay_pos = (self._replay_pos + 1) % self.replay_capacity
        else:
            self.replay.append(
                _Transition(
                    state,
                    action,
                    float(reward),
                    next_state,
                    next_action_mask,
                    done,
                )
            )

    def train_step(self) -> float | None:
        if len(self.replay) < max(self.batch_size, self.warmup_steps):
            return None
        
        # --- PER Sampling ---
        if self.use_per:
            # Anneal beta from per_beta_start to 1.0 over per_beta_frames
            self._per_frame += 1
            beta_progress = min(self._per_frame / self.per_beta_frames, 1.0)
            self.per_beta = self.per_beta_start + (1.0 - self.per_beta_start) * beta_progress
            
            # Sample proportional to priority^alpha
            priorities = self._replay_priorities[:self._replay_filled] ** self.per_alpha
            probs = priorities / priorities.sum()
            
            # Sample indices
            indices = self._rng.choices(
                range(self._replay_filled), 
                weights=probs, 
                k=self.batch_size
            )
            batch = [self.replay[i] for i in indices]
            
            # Importance sampling weights: (N * P(i))^(-beta)
            weights = (self._replay_filled * np.array([probs[i] for i in indices])) ** (-self.per_beta)
            weights = weights / weights.max()  # Normalize for stability
            is_weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
        else:
            batch = self._rng.sample(self.replay, self.batch_size)
            is_weights = torch.ones(self.batch_size, dtype=torch.float32, device=self.device)
            indices = None
        
        states = torch.tensor(
            np.asarray([item.state for item in batch]),
            dtype=torch.float32,
            device=self.device,
        )
        actions = torch.tensor(
            [item.action for item in batch],
            dtype=torch.int64,
            device=self.device,
        )
        rewards = torch.tensor(
            [item.reward for item in batch],
            dtype=torch.float32,
            device=self.device,
        )
        next_states = torch.tensor(
            np.asarray([item.next_state for item in batch]),
            dtype=torch.float32,
            device=self.device,
        )
        next_masks = torch.tensor(
            np.asarray([item.next_action_mask for item in batch]),
            dtype=torch.bool,
            device=self.device,
        )
        dones = torch.tensor([item.done for item in batch], dtype=torch.float32, device=self.device)

        # --- Main DQN Loss (Double DQN) ---
        current_q = self.online.get_q_values(states)
        current_values = current_q.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            online_next_q = self.online.get_q_values(next_states).masked_fill(
                ~next_masks,
                -torch.inf,
            )
            next_actions = online_next_q.argmax(dim=1)
            target_next_q = self.target.get_q_values(next_states).gather(
                1,
                next_actions.unsqueeze(1),
            ).squeeze(1)
            target_next_q = torch.where(dones.bool(), 0.0, target_next_q)
            expected_values = rewards + self.gamma * target_next_q

        # Per-sample TD errors for priority update
        td_errors = (current_values - expected_values).abs().detach()
        
        # Weighted loss with importance sampling
        dqn_loss = (is_weights * nn.functional.smooth_l1_loss(current_values, expected_values, reduction='none')).mean()
        
        # --- Auxiliary Losses (Multi-task Learning) ---
        aux_loss = torch.tensor(0.0, device=self.device)
        if self.aux_loss_weight > 0:
            # Get all heads output
            online_out = self.online.forward_all(states)
            
            # 1. Reward prediction (aux head 0)
            pred_reward = online_out["aux"][0]
            aux_loss += nn.functional.mse_loss(pred_reward, rewards)
            
            # 2. Next-state feature prediction (aux head 1) - representation learning
            # Predict next state's features (simplified: predict mean of next state)
            target_next_features = next_states.mean(dim=1)  # (B,)
            pred_next_features = online_out["aux"][1]
            aux_loss += nn.functional.mse_loss(pred_next_features, target_next_features)
            
            # 3. Uncertainty estimation (aux head 2) - predict TD error magnitude
            pred_uncertainty = online_out["aux"][2]
            aux_loss += nn.functional.mse_loss(pred_uncertainty, td_errors)
            
            aux_loss = aux_loss / 3.0  # Average
        
        # --- Policy Distillation Loss (match policy head to Q-head softmax) ---
        policy_loss = torch.tensor(0.0, device=self.device)
        if self.policy_distill_weight > 0:
            policy_logits = self.online.get_policy_logits(states)
            with torch.no_grad():
                # Soft targets from Q-values
                q_targets = current_q / 0.1  # temperature
                policy_targets = torch.softmax(q_targets, dim=-1)
            policy_loss = nn.functional.cross_entropy(
                policy_logits, 
                policy_targets.argmax(dim=-1)
            )
        
        # --- Combined Loss ---
        total_loss = dqn_loss + self.aux_loss_weight * aux_loss + self.policy_distill_weight * policy_loss
        
        # --- Update PER priorities ---
        if self.use_per and indices is not None:
            # New priorities = |TD error| + eps
            new_priorities = td_errors.cpu().numpy() + self.per_eps
            for idx, new_prio in zip(indices, new_priorities):
                self._replay_priorities[idx] = new_prio
        
        if self._optimizer_on_cpu:
            # CPU Optimizer Pattern: Gradients computed on GPU, optimizer step on CPU
            # 1. Zero CPU optimizer gradients
            self.optimizer.zero_grad()
            
            # 2. Backward on GPU (gradients accumulate on GPU params)
            total_loss.backward()
            
            # 3. Copy gradients from GPU params to CPU params
            with torch.no_grad():
                for gpu_p, cpu_p in zip(self._gpu_params, self.optimizer.param_groups[0]['params']):
                    if gpu_p.grad is not None:
                        cpu_p.grad = gpu_p.grad.detach().cpu()
            
            # 4. Clip gradient norm on CPU params
            nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]['params'], max_norm=1.0)
            
            # 5. Optimizer step on CPU
            self.optimizer.step()
            
            # 6. Copy updated CPU params back to GPU params
            with torch.no_grad():
                for gpu_p, cpu_p in zip(self._gpu_params, self.optimizer.param_groups[0]['params']):
                    gpu_p.copy_(cpu_p)
            
            # 7. Clear GPU gradients
            for gpu_p in self._gpu_params:
                gpu_p.grad = None
        else:
            # Standard path (CPU or CUDA)
            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=1.0)
            self.optimizer.step()
        
        self.training_steps += 1
        if self.training_steps % self.target_update_steps == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(total_loss.item())

    def behavior_clone(
        self,
        samples: list[tuple[tuple[float, ...], int, tuple[bool, ...]]],
        epochs: int = 5,
        batch_size: int = 128,
    ) -> float:
        if not samples:
            raise ValueError("Behavior cloning requires at least one demonstration.")
        if epochs <= 0 or batch_size <= 0:
            raise ValueError("Behavior-cloning epochs and batch size must be positive.")
        for state, action, action_mask in samples:
            self._validate_state(state)
            if (
                not 0 <= action < self.action_count
                or len(action_mask) != self.action_count
                or not action_mask[action]
            ):
                raise ValueError("Demonstration action is invalid for its action mask.")

        losses: list[float] = []
        effective_batch_size = min(batch_size, len(samples))
        for _ in range(epochs):
            shuffled = self._rng.sample(samples, len(samples))
            for start in range(0, len(shuffled), effective_batch_size):
                batch = shuffled[start : start + effective_batch_size]
                states = torch.tensor(
                    np.asarray([sample[0] for sample in batch]),
                    dtype=torch.float32,
                    device=self.device,
                )
                actions = torch.tensor(
                    [sample[1] for sample in batch],
                    dtype=torch.int64,
                    device=self.device,
                )
                action_masks = torch.tensor(
                    np.asarray([sample[2] for sample in batch]),
                    dtype=torch.bool,
                    device=self.device,
                )
                # Use policy head for behavior cloning (designed for probability matching)
                logits = self.online.get_policy_logits(states).masked_fill(
                    ~action_masks,
                    -torch.inf,
                )
                loss = nn.functional.cross_entropy(logits, actions)
                
                if self._optimizer_on_cpu:
                    # CPU Optimizer Pattern
                    self.optimizer.zero_grad()
                    loss.backward()
                    with torch.no_grad():
                        for gpu_p, cpu_p in zip(self._gpu_params, self.optimizer.param_groups[0]['params']):
                            if gpu_p.grad is not None:
                                cpu_p.grad = gpu_p.grad.detach().cpu()
                    nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]['params'], max_norm=1.0)
                    self.optimizer.step()
                    with torch.no_grad():
                        for gpu_p, cpu_p in zip(self._gpu_params, self.optimizer.param_groups[0]['params']):
                            gpu_p.copy_(cpu_p)
                    for gpu_p in self._gpu_params:
                        gpu_p.grad = None
                else:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=1.0)
                    self.optimizer.step()
                
                self.training_steps += 1
                losses.append(float(loss.item()))
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        return float(np.mean(losses))

    def save(
        self,
        path: Path,
        metadata: dict[str, Any] | None = None,
        include_training_state: bool = True,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Extract architecture config from online network
        arch_config = {
            "hidden_size": self.online.hidden_size,
            "num_layers": getattr(self.online, 'num_layers', 3),
            "use_spatial": self.online.use_spatial,
            "spatial_shape": self.online.spatial_shape,
            "num_aux_heads": self.online.num_aux_heads,
            "dropout": 0.1,  # Not stored in module, use default
        }
        payload: dict[str, Any] = {
            "format_version": self.FORMAT_VERSION,
            "input_size": self.input_size,
            "action_count": self.action_count,
            "architecture_config": arch_config,
            "online_state_dict": self.online.state_dict(),
            "target_state_dict": self.target.state_dict(),
            "environment_steps": self.environment_steps,
            "training_steps": self.training_steps,
            "metadata": metadata or {},
            "has_training_state": include_training_state,
        }
        if include_training_state:
            # Handle PER vs regular replay
            if self.use_per:
                replay_data = [
                    {
                        "state": item.state,
                        "action": item.action,
                        "reward": item.reward,
                        "next_state": item.next_state,
                        "next_action_mask": item.next_action_mask,
                        "done": item.done,
                        "priority": item.priority,
                    }
                    for item in self.replay
                ]
                replay_capacity = self.replay_capacity
                per_state = {
                    "priorities": self._replay_priorities[:self._replay_filled].tolist(),
                    "replay_pos": self._replay_pos,
                    "replay_filled": self._replay_filled,
                    "per_frame": self._per_frame,
                    "per_beta": self.per_beta,
                }
            else:
                replay_data = [
                    (
                        item.state,
                        item.action,
                        item.reward,
                        item.next_state,
                        item.next_action_mask,
                        item.done,
                    )
                    for item in self.replay
                ]
                replay_capacity = self.replay.maxlen
                per_state = None
            
            payload.update(
                {
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "replay_capacity": replay_capacity,
                    "replay": replay_data,
                    "use_per": self.use_per,
                    "per_state": per_state,
                    "python_rng_state": self._rng.getstate(),
                    "training_config": {
                        "batch_size": self.batch_size,
                        "gamma": self.gamma,
                        "epsilon_decay_steps": self.epsilon_decay_steps,
                        "epsilon_start": self.epsilon_start,
                        "epsilon_end": self.epsilon_end,
                        "warmup_steps": self.warmup_steps,
                        "target_update_steps": self.target_update_steps,
                        "aux_loss_weight": self.aux_loss_weight,
                        "policy_distill_weight": self.policy_distill_weight,
                        "per_alpha": self.per_alpha,
                        "per_beta_start": self.per_beta_start,
                        "per_beta_frames": self.per_beta_frames,
                        "per_eps": self.per_eps,
                    },
                }
            )
        temporary_path = path.with_name(f"{path.name}.tmp")
        torch.save(payload, temporary_path)
        temporary_path.replace(path)

    def load(self, path: Path, resume_training: bool = True) -> dict[str, Any]:
        # Load to CPU first, then move to target device
        # DirectML doesn't support direct map_location to privateuseone device
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if (
            not isinstance(payload, dict)
            or payload.get("input_size") != self.input_size
            or payload.get("action_count") != self.action_count
            or not isinstance(payload.get("online_state_dict"), dict)
        ):
            raise ValueError(f"Incompatible neural policy checkpoint: {path}")
        if payload.get("format_version") == 1:
            if resume_training:
                raise ValueError(
                    "This legacy checkpoint contains policy weights only. "
                    "Choose a new --model path with --fresh to begin resumable training."
                )
            self.online.load_state_dict(payload["online_state_dict"], strict=True)
            self.target.load_state_dict(self.online.state_dict())
            self.target.eval()
            self.environment_steps = 0
            self.training_steps = 0
            return {}
        if (
            payload.get("format_version") != self.FORMAT_VERSION
            or not isinstance(payload.get("target_state_dict"), dict)
        ):
            raise ValueError(f"Incompatible neural policy checkpoint: {path}")
        
        # Check architecture compatibility (only critical fields)
        saved_arch = payload.get("architecture_config", {})
        current_arch = {
            "hidden_size": self.online.hidden_size,
            "num_layers": getattr(self.online, 'num_layers', 3),
            "use_spatial": self.online.use_spatial,
            "spatial_shape": self.online.spatial_shape,
            "num_aux_heads": self.online.num_aux_heads,
            "dropout": getattr(self.online, 'dropout', 0.1),
        }
        # Allow loading if architecture matches (or if not saved, assume compatible)
        # Compare only critical fields that affect weight shapes
        critical_fields = ["hidden_size", "use_spatial", "spatial_shape", "num_aux_heads"]
        if saved_arch:
            mismatch = any(
                saved_arch.get(k) != current_arch.get(k) 
                for k in critical_fields
            )
            if mismatch:
                raise ValueError(
                    f"Architecture mismatch on critical fields: checkpoint={saved_arch}, current={current_arch}. "
                    "Create new agent with matching architecture or use --fresh."
                )
        
        self.online.load_state_dict(payload["online_state_dict"], strict=True)
        self.target.load_state_dict(payload["target_state_dict"], strict=True)
        self.target.eval()
        self.environment_steps = int(payload.get("environment_steps", 0))
        self.training_steps = int(payload.get("training_steps", 0))
        if resume_training and not payload.get("has_training_state", False):
            raise ValueError(
                f"Checkpoint contains policy weights only, not resumable training state: {path}"
            )
        if resume_training:
            required_state = {
                "optimizer_state_dict",
                "replay_capacity",
                "replay",
                "python_rng_state",
                "training_config",
            }
            if not required_state.issubset(payload):
                raise ValueError(f"Incomplete DQN training state in checkpoint: {path}")
            config = payload.get("training_config")
            if not isinstance(config, dict):
                raise ValueError(f"Invalid DQN training configuration in checkpoint: {path}")
            normalized_config = {
                **config,
                "epsilon_start": config.get("epsilon_start", 1.0),
                "epsilon_end": config.get("epsilon_end", 0.05),
                "aux_loss_weight": config.get("aux_loss_weight", 0.1),
                "policy_distill_weight": config.get("policy_distill_weight", 0.05),
                "per_alpha": config.get("per_alpha", 0.6),
                "per_beta_start": config.get("per_beta_start", 0.4),
                "per_beta_frames": config.get("per_beta_frames", 100_000),
                "per_eps": config.get("per_eps", 1e-6),
            }
            expected_config = {
                "batch_size": self.batch_size,
                "gamma": self.gamma,
                "epsilon_decay_steps": self.epsilon_decay_steps,
                "epsilon_start": self.epsilon_start,
                "epsilon_end": self.epsilon_end,
                "warmup_steps": self.warmup_steps,
                "target_update_steps": self.target_update_steps,
                "aux_loss_weight": self.aux_loss_weight,
                "policy_distill_weight": self.policy_distill_weight,
                "per_alpha": self.per_alpha,
                "per_beta_start": self.per_beta_start,
                "per_beta_frames": self.per_beta_frames,
                "per_eps": self.per_eps,
            }
            if normalized_config != expected_config:
                raise ValueError(
                    "Checkpoint training configuration does not match this DQN agent."
                )
            replay_capacity = payload.get("replay_capacity")
            replay = payload.get("replay")
            rng_state = payload.get("python_rng_state")
            optimizer_state = payload.get("optimizer_state_dict")
            use_per = payload.get("use_per", False)
            per_state = payload.get("per_state")
            
            if (
                not isinstance(replay_capacity, int)
                or replay_capacity <= 0
                or not isinstance(replay, list)
                or not isinstance(rng_state, tuple)
                or not isinstance(optimizer_state, dict)
            ):
                raise ValueError(f"Invalid DQN training state in checkpoint: {path}")
            
            if use_per != self.use_per:
                raise ValueError(
                    f"PER setting mismatch: checkpoint use_per={use_per}, current use_per={self.use_per}"
                )
            
            if self.use_per:
                # Restore PER replay buffer
                self.replay = []
                self._replay_priorities = np.zeros(replay_capacity, dtype=np.float32)
                self._replay_pos = 0
                self._replay_filled = 0
                
                for item in replay:
                    if not isinstance(item, dict):
                        raise ValueError(f"Invalid PER replay transition in checkpoint: {path}")
                    transition = _PrioritizedTransition(
                        state=tuple(item["state"]),
                        action=int(item["action"]),
                        reward=float(item["reward"]),
                        next_state=tuple(item["next_state"]),
                        next_action_mask=tuple(item["next_action_mask"]),
                        done=bool(item["done"]),
                        priority=float(item.get("priority", 1.0)),
                        idx=self._replay_pos,
                    )
                    self.replay.append(transition)
                    self._replay_priorities[self._replay_pos] = transition.priority
                    self._replay_pos = (self._replay_pos + 1) % replay_capacity
                    if self._replay_filled < replay_capacity:
                        self._replay_filled += 1
                
                # Restore PER state
                if per_state:
                    self._replay_priorities[:self._replay_filled] = np.array(per_state["priorities"], dtype=np.float32)
                    self._replay_pos = per_state.get("replay_pos", 0)
                    self._replay_filled = per_state.get("replay_filled", self._replay_filled)
                    self._per_frame = per_state.get("per_frame", 0)
                    self.per_beta = per_state.get("per_beta", self.per_beta_start)
            else:
                # Regular replay buffer
                if replay_capacity != self.replay.maxlen:
                    raise ValueError(f"Invalid DQN training state in checkpoint: {path}")
                self.replay = deque(maxlen=replay_capacity)
                for transition in replay:
                    if not isinstance(transition, tuple) or len(transition) != 6:
                        raise ValueError(f"Invalid replay transition in checkpoint: {path}")
                    state, action, reward, next_state, action_mask, done = transition
                    self.remember(
                        tuple(state),
                        int(action),
                        float(reward),
                        tuple(next_state),
                        tuple(action_mask),
                        bool(done),
                    )
            
            self.optimizer.load_state_dict(optimizer_state)
            self._rng.setstate(rng_state)
        return payload.get("metadata", {})

    def _validate_state(self, state: tuple[float, ...]) -> None:
        if len(state) != self.input_size or any(
            not math.isfinite(float(value)) for value in state
        ):
            raise ValueError("State must have input_size finite numeric features.")
