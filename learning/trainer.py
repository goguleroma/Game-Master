"""
Unified Training Loop with Modern RL Improvements.

Features:
- Conservative Q-Learning (CQL) regularization
- Mixed Precision (fp16/bf16) with GradScaler
- Prioritized Experience Replay (improved PER)
- Larger batch sizes with gradient accumulation
- EMA for evaluation
- torch.compile support
- Distributional RL (C51) support
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple, List
import math
import random
from collections import deque

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from learning.networks import (
    ModernQNetwork,
    ModernQNetworkConfig,
    EMAModel,
    categorical_projection,
    create_modern_q_network,
)
from learning.world_model import WorldModel, WorldModelConfig, create_world_model


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class TrainerConfig:
    # Network
    input_dim: int
    action_dim: int
    hidden_dim: int = 512
    num_residual_blocks: int = 6
    use_moe: bool = True
    use_distributional: bool = True
    num_atoms: int = 51
    v_min: float = -10.0
    v_max: float = 10.0
    use_spectral_norm: bool = True
    dropout: float = 0.1
    
    # World Model
    use_world_model: bool = True
    deter_dim: int = 512
    num_categoricals: int = 32
    num_classes: int = 32
    codebook_dim: int = 256
    wm_hidden_dim: int = 512
    wm_num_layers: int = 3
    wm_dropout: float = 0.1
    wm_ensemble_size: int = 5
    imagination_horizon: int = 15
    
    # Training
    batch_size: int = 256
    gradient_accumulation_steps: int = 2  # Effective batch = 512
    gamma: float = 0.99
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    
    # CQL
    cql_alpha: float = 1.0
    cql_temp: float = 1.0
    cql_min_q_weight: float = 5.0
    
    # PER
    use_per: bool = True
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_frames: int = 200_000
    per_eps: float = 1e-6  # Maps to PrioritizedReplayBuffer.eps
    replay_capacity: int = 100_000
    
    # EMA
    ema_decay: float = 0.9995
    ema_update_after: int = 1000
    ema_update_every: int = 1
    
    # Mixed Precision
    use_mixed_precision: bool = True
    precision: str = "bf16"  # "fp16" or "bf16"
    
    # Optimization
    target_update_steps: int = 1000
    warmup_steps: int = 1000
    
    # Exploration
    epsilon_decay_steps: int = 50_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    
    # Compilation
    compile_model: bool = True
    compile_mode: str = "reduce-overhead"  # "default", "reduce-overhead", "max-autotune"
    
    # Logging
    log_interval: int = 100
    eval_interval: int = 1000
    save_interval: int = 5000


# ============================================================================
# Improved Prioritized Replay Buffer
# ============================================================================

class PrioritizedReplayBuffer:
    """
    Improved PER with segment tree for O(log n) sampling.
    """
    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 200_000,
        eps: float = 1e-6,
        seed: int = 0,
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.eps = eps
        self._rng = random.Random(seed)
        self._beta = beta_start
        self._frame = 0
        
        # Segment tree for priorities
        self._tree_size = 1
        while self._tree_size < capacity:
            self._tree_size *= 2
        self._tree = np.zeros(2 * self._tree_size, dtype=np.float32)
        self._max_priority = 1.0
        
        # Data storage
        self._data = [None] * capacity
        self._pos = 0
        self._size = 0
    
    @property
    def beta(self) -> float:
        return self._beta
    
    def _update_beta(self) -> None:
        self._frame += 1
        progress = min(self._frame / self.beta_frames, 1.0)
        self._beta = self.beta_start + (1.0 - self.beta_start) * progress
    
    def _propagate(self, idx: int, change: float) -> None:
        parent = idx // 2
        while parent >= 1:
            self._tree[parent] += change
            parent //= 2
    
    def _set_priority(self, idx: int, priority: float) -> None:
        tree_idx = idx + self._tree_size
        change = priority - self._tree[tree_idx]
        self._tree[tree_idx] = priority
        self._propagate(tree_idx, change)
    
    def _get_priority(self, idx: int) -> float:
        return self._tree[idx + self._tree_size]
    
    def _sum(self) -> float:
        return self._tree[1]
    
    def _sample_index(self, value: float) -> int:
        idx = 1
        while idx < self._tree_size:
            left = idx * 2
            if value <= self._tree[left]:
                idx = left
            else:
                value -= self._tree[left]
                idx = left + 1
        return idx - self._tree_size
    
    def add(self, transition: tuple, priority: Optional[float] = None) -> None:
        """Add transition with priority."""
        if priority is None:
            priority = self._max_priority
        
        self._data[self._pos] = transition
        self._set_priority(self._pos, priority ** self.alpha)
        self._max_priority = max(self._max_priority, priority)
        
        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple[List[tuple], np.ndarray, np.ndarray]:
        """Sample batch with importance sampling weights."""
        self._update_beta()
        
        if self._size < batch_size:
            raise ValueError("Not enough samples in buffer")
        
        # Sample using segment tree
        segment = self._sum() / batch_size
        indices = []
        transitions = []
        
        for i in range(batch_size):
            value = self._rng.uniform(i * segment, (i + 1) * segment)
            idx = self._sample_index(value)
            indices.append(idx)
            transitions.append(self._data[idx])
        
        # Importance sampling weights
        priorities = np.array([self._get_priority(idx) for idx in indices])
        probs = priorities / self._sum()
        weights = (self._size * probs) ** (-self._beta)
        weights = weights / weights.max()  # Normalize
        
        return transitions, indices, weights.astype(np.float32)
    
    def update_priorities(self, indices: List[int], priorities: np.ndarray) -> None:
        """Update priorities for sampled transitions."""
        for idx, priority in zip(indices, priorities):
            priority = float(priority) + self.eps
            self._set_priority(idx, priority ** self.alpha)
            self._max_priority = max(self._max_priority, priority)
    
    def __len__(self) -> int:
        return self._size


# ============================================================================
# Transition Dataclass
# ============================================================================

@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    next_action_mask: np.ndarray
    done: bool


# ============================================================================
# Conservative Q-Learning (CQL) Loss
# ============================================================================

def cql_loss(
    q_values: torch.Tensor,
    actions: torch.Tensor,
    q_target: torch.Tensor,
    alpha: float = 1.0,
    temp: float = 1.0,
    min_q_weight: float = 5.0,
) -> torch.Tensor:
    """
    Conservative Q-Learning regularization.
    
    CQL loss = alpha * (logsumexp(Q) - Q(a)) + min_q_weight * (Q(a) - Q_target)^2
    
    Args:
        q_values: (batch, action_dim) - Q-values for all actions
        actions: (batch,) - taken actions
        q_target: (batch,) - target Q-values
        alpha: CQL regularization weight
        temp: Temperature for logsumexp
        min_q_weight: Weight for min Q regularization
    """
    batch_size = q_values.shape[0]
    
    # Current Q for taken actions
    q_a = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    
    # LogSumExp over all actions (with temperature)
    logsumexp_q = torch.logsumexp(q_values / temp, dim=1) * temp
    
    # CQL regularization: push down Q-values for unseen actions
    cql_reg = alpha * (logsumexp_q - q_a).mean()
    
    # Min Q regularization: push Q(a) towards target
    min_q_reg = min_q_weight * F.mse_loss(q_a, q_target.detach())
    
    return cql_reg + min_q_reg


# ============================================================================
# C51 Distributional Loss
# ============================================================================

def c51_loss(
    logits: torch.Tensor,
    target_logits: torch.Tensor,
    supports: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """
    C51 Categorical DQN loss (cross-entropy with projected distribution).
    
    Args:
        logits: (batch, action_dim, num_atoms) - online network logits
        target_logits: (batch, action_dim, num_atoms) - target network logits
        supports: (num_atoms,) - value atoms
        actions: (batch,) - actions taken
        rewards: (batch,) - rewards
        dones: (batch,) - done flags
        gamma: discount factor
    """
    batch_size, action_dim, num_atoms = logits.shape
    
    # Get target distribution for next states
    with torch.no_grad():
        # Double DQN: use online network to select actions
        target_q = (F.softmax(target_logits, dim=-1) * supports).sum(-1)
        next_actions = target_q.argmax(dim=1)
        
        # Get target distribution for selected actions
        target_dist = F.softmax(target_logits, dim=-1)
        next_dist = target_dist[torch.arange(batch_size), next_actions]
        
        # Project onto current supports
        target_supports = rewards.view(-1, 1) + gamma * supports.view(1, -1) * (~dones).view(-1, 1).float()
        target_supports = target_supports.clamp(supports[0].item(), supports[-1].item())
        
        # Compute projection
        delta_z = (supports[-1] - supports[0]) / (num_atoms - 1)
        b = (target_supports - supports[0]) / delta_z
        l = b.floor().long().clamp(0, num_atoms - 1)
        u = b.ceil().long().clamp(0, num_atoms - 1)
        
        # Distribute probability mass
        m = torch.zeros_like(next_dist)
        offset = torch.arange(batch_size, device=logits.device).view(-1, 1) * num_atoms
        
        l_flat = (l + offset).view(-1)
        u_flat = (u + offset).view(-1)
        next_dist_flat = next_dist.view(-1)
        
        w_l = (u.float() - b).view(-1)
        w_u = (b - l.float()).view(-1)
        
        m_flat = m.view(-1)
        m_flat.index_add_(0, l_flat, next_dist_flat * w_l)
        m_flat.index_add_(0, u_flat, next_dist_flat * w_u)
    
    # Cross-entropy loss
    log_probs = F.log_softmax(logits[torch.arange(batch_size), actions], dim=-1)
    loss = -(m * log_probs).sum(-1).mean()
    
    return loss


# ============================================================================
# Unified Trainer
# ============================================================================

class Trainer:
    """
    Unified training loop for DQN + World Model.
    
    Supports:
    - Standard DQN / Double DQN
    - C51 Distributional RL
    - CQL regularization
    - Model-based imagination (World Model)
    - Mixed precision training
    - EMA for evaluation
    - Gradient accumulation
    """
    def __init__(
        self,
        config: TrainerConfig,
        device: Optional[torch.device] = None,
        seed: int = 0,
    ):
        self.config = config
        self.seed = seed
        self._rng = random.Random(seed)
        self._torch_generator = torch.Generator()
        self._torch_generator.manual_seed(seed)
        
        # Device
        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        print(f"Trainer using device: {self.device}")
        
        # Precision setup
        self.use_amp = config.use_mixed_precision and self.device.type == "cuda"
        if self.use_amp:
            if config.precision == "bf16" and torch.cuda.is_bf16_supported():
                self.amp_dtype = torch.bfloat16
            else:
                self.amp_dtype = torch.float16
            self.scaler = GradScaler()
            print(f"Mixed precision enabled: {self.amp_dtype}")
        else:
            self.amp_dtype = torch.float32
            self.scaler = None
        
        # Networks
        self._build_networks()
        
        # Optimizer
        self._build_optimizer()
        
        # Replay buffer
        self.replay = PrioritizedReplayBuffer(
            capacity=config.replay_capacity,
            alpha=config.per_alpha,
            beta_start=config.per_beta_start,
            beta_frames=config.per_beta_frames,
            eps=config.per_eps,
            seed=seed,
        ) if config.use_per else None
        
        # Simple replay fallback
        self._simple_replay = deque(maxlen=config.replay_capacity) if not config.use_per else None
        
        # Training state
        self.environment_steps = 0
        self.training_steps = 0
        self.optimizer_steps = 0
        self.accumulation_step = 0
        
        # Compile models
        if config.compile_model and hasattr(torch, 'compile'):
            self._compile_models()
    
    def _build_networks(self) -> None:
        """Build all networks."""
        # Q-Networks
        q_config = ModernQNetworkConfig(
            input_dim=self.config.input_dim,
            action_dim=self.config.action_dim,
            hidden_dim=self.config.hidden_dim,
            num_residual_blocks=self.config.num_residual_blocks,
            use_moe=self.config.use_moe,
            use_distributional=self.config.use_distributional,
            num_atoms=self.config.num_atoms,
            v_min=self.config.v_min,
            v_max=self.config.v_max,
            use_spectral_norm=self.config.use_spectral_norm,
            dropout=self.config.dropout,
        )
        
        self.online_net = create_modern_q_network(
            input_dim=self.config.input_dim,
            action_dim=self.config.action_dim,
            hidden_dim=self.config.hidden_dim,
            num_residual_blocks=self.config.num_residual_blocks,
            use_moe=self.config.use_moe,
            use_distributional=self.config.use_distributional,
            num_atoms=self.config.num_atoms,
            use_spectral_norm=self.config.use_spectral_norm,
            dropout=self.config.dropout,
        ).to(self.device)
        
        self.target_net = create_modern_q_network(
            input_dim=self.config.input_dim,
            action_dim=self.config.action_dim,
            hidden_dim=self.config.hidden_dim,
            num_residual_blocks=self.config.num_residual_blocks,
            use_moe=self.config.use_moe,
            use_distributional=self.config.use_distributional,
            num_atoms=self.config.num_atoms,
            use_spectral_norm=self.config.use_spectral_norm,
            dropout=self.config.dropout,
        ).to(self.device)
        
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        
        # EMA
        self.ema = EMAModel(
            self.online_net,
            decay=self.config.ema_decay,
            update_after_step=self.config.ema_update_after,
            update_every=self.config.ema_update_every,
        ).to(self.device)
        
        # World Model
        if self.config.use_world_model:
            self.world_model = create_world_model(
                obs_dim=self.config.input_dim,
                action_dim=self.config.action_dim,
                deter_dim=self.config.deter_dim,
                num_categoricals=self.config.num_categoricals,
                num_classes=self.config.num_classes,
                codebook_dim=self.config.codebook_dim,
                hidden_dim=self.config.wm_hidden_dim,
                num_layers=self.config.wm_num_layers,
                dropout=self.config.wm_dropout,
                use_spectral_norm=self.config.use_spectral_norm,
                ensemble_size=self.config.wm_ensemble_size,
                imagination_horizon=self.config.imagination_horizon,
            ).to(self.device)
        else:
            self.world_model = None
    
    def _build_optimizer(self) -> None:
        """Build optimizer with parameter groups."""
        # Separate parameter groups for different learning rates if needed
        params = [
            {'params': self.online_net.parameters(), 'lr': self.config.learning_rate, 'weight_decay': self.config.weight_decay},
        ]
        
        if self.world_model is not None:
            params.append({
                'params': self.world_model.parameters(),
                'lr': self.config.learning_rate,
                'weight_decay': self.config.weight_decay,
            })
        
        self.optimizer = torch.optim.AdamW(params, fused=True if self.device.type == "cuda" else False)
    
    def _compile_models(self) -> None:
        """Compile models with torch.compile."""
        print("Compiling models...")
        self.online_net = torch.compile(self.online_net, mode=self.config.compile_mode)
        self.target_net = torch.compile(self.target_net, mode=self.config.compile_mode)
        if self.world_model is not None:
            self.world_model = torch.compile(self.world_model, mode=self.config.compile_mode)
        print("Models compiled.")
    
    @property
    def epsilon(self) -> float:
        progress = min(self.environment_steps / self.config.epsilon_decay_steps, 1.0)
        return self.config.epsilon_start - (self.config.epsilon_start - self.config.epsilon_end) * progress
    
    def select_action(
        self,
        state: np.ndarray,
        action_mask: np.ndarray,
        explore: bool = True,
        use_ema: bool = False,
    ) -> int:
        """Select action using current policy."""
        net = self.ema.model if use_ema and self.ema.step > self.config.ema_update_after else self.online_net
        
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            mask_tensor = torch.from_numpy(action_mask).bool().unsqueeze(0).to(self.device)
            
            if self.use_amp:
                with autocast(dtype=self.amp_dtype):
                    q_values = net.get_q_values(state_tensor)
            else:
                q_values = net.get_q_values(state_tensor)
            
            q_values = q_values.masked_fill(~mask_tensor, -torch.inf)
            
            if explore and self._rng.random() < self.epsilon:
                valid_actions = torch.where(mask_tensor[0])[0]
                action = int(self._rng.choice(valid_actions.cpu().numpy()))
            else:
                action = int(q_values.argmax(dim=1).item())
        
        if explore:
            self.environment_steps += 1
        
        return action
    
    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        next_action_mask: np.ndarray,
        done: bool,
    ) -> None:
        """Store transition in replay buffer."""
        transition = Transition(
            state=state.astype(np.float32),
            action=action,
            reward=float(reward),
            next_state=next_state.astype(np.float32),
            next_action_mask=next_action_mask.astype(np.bool_),
            done=done,
        )
        
        if self.config.use_per:
            self.replay.add(transition)
        else:
            self._simple_replay.append(transition)
    
    def train_step(self) -> Optional[dict]:
        """Single training step with gradient accumulation."""
        # Check if enough data
        min_size = max(self.config.batch_size, self.config.warmup_steps)
        current_size = len(self.replay) if self.config.use_per else len(self._simple_replay)
        if current_size < min_size:
            return None
        
        # Sample batch
        if self.config.use_per:
            transitions, indices, weights = self.replay.sample(self.config.batch_size)
            is_weights = torch.from_numpy(weights).to(self.device)
        else:
            transitions = self._rng.sample(self._simple_replay, self.config.batch_size)
            indices = None
            is_weights = torch.ones(self.config.batch_size, device=self.device)
        
        # Prepare tensors
        states = torch.from_numpy(np.stack([t.state for t in transitions])).to(self.device)
        actions = torch.tensor([t.action for t in transitions], dtype=torch.long, device=self.device)
        rewards = torch.tensor([t.reward for t in transitions], dtype=torch.float32, device=self.device)
        next_states = torch.from_numpy(np.stack([t.next_state for t in transitions])).to(self.device)
        next_masks = torch.from_numpy(np.stack([t.next_action_mask for t in transitions])).to(self.device)
        dones = torch.tensor([t.done for t in transitions], dtype=torch.float32, device=self.device)
        
        # Forward pass with mixed precision
        if self.use_amp:
            with autocast(dtype=self.amp_dtype):
                loss_dict = self._compute_loss(
                    states, actions, rewards, next_states, next_masks, dones, is_weights
                )
                loss = loss_dict['total_loss'] / self.config.gradient_accumulation_steps
            
            # Backward
            self.scaler.scale(loss).backward()
        else:
            loss_dict = self._compute_loss(
                states, actions, rewards, next_states, next_masks, dones, is_weights
            )
            loss = loss_dict['total_loss'] / self.config.gradient_accumulation_steps
            loss.backward()
        
        self.accumulation_step += 1
        
        # Optimizer step
        if self.accumulation_step >= self.config.gradient_accumulation_steps:
            if self.use_amp:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.online_net.parameters(), self.config.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(
                    self.online_net.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()
            
            self.optimizer.zero_grad()
            self.accumulation_step = 0
            self.optimizer_steps += 1
            
            # Update target network
            if self.optimizer_steps % self.config.target_update_steps == 0:
                self.target_net.load_state_dict(self.online_net.state_dict())
            
            # Update EMA
            self.ema.update()
            
            self.training_steps += 1
        
        # Update PER priorities
        if self.config.use_per and indices is not None and 'td_errors' in loss_dict:
            self.replay.update_priorities(indices, loss_dict['td_errors'].cpu().numpy())
        
        return loss_dict
    
    def _compute_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        next_masks: torch.Tensor,
        dones: torch.Tensor,
        is_weights: torch.Tensor,
    ) -> dict:
        """Compute all losses."""
        batch_size = states.shape[0]
        losses = {}
        
        # === Main DQN Loss ===
        if self.config.use_distributional:
            # C51 Loss
            online_logits = self.online_net.get_q_logits(states)
            with torch.no_grad():
                target_logits = self.target_net.get_q_logits(next_states)
            
            # Double DQN action selection
            target_q = (F.softmax(target_logits, dim=-1) * self.target_net.q_head.supports).sum(-1)
            target_q = target_q.masked_fill(~next_masks, -torch.inf)
            next_actions = target_q.argmax(dim=1)
            
            dqn_loss = c51_loss(
                online_logits,
                target_logits,
                self.online_net.q_head.supports,
                actions,
                rewards,
                dones.bool(),
                self.config.gamma,
            )
            
            # TD errors for PER (use Q-value difference)
            with torch.no_grad():
                online_q = self.online_net.get_q_values(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                target_q_values = self.target_net.get_q_values(next_states)
                target_q_values = target_q_values.masked_fill(~next_masks, -torch.inf)
                next_q = target_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)
                next_q = torch.where(dones.bool(), 0.0, next_q)
                expected_q = rewards + self.config.gamma * next_q
                td_errors = (online_q - expected_q).abs()
            
            losses['dqn_loss'] = dqn_loss
            
        else:
            # Standard Double DQN
            current_q = self.online_net.get_q_values(states)
            current_values = current_q.gather(1, actions.unsqueeze(1)).squeeze(1)
            
            with torch.no_grad():
                online_next_q = self.online_net.get_q_values(next_states).masked_fill(~next_masks, -torch.inf)
                next_actions = online_next_q.argmax(dim=1)
                target_next_q = self.target_net.get_q_values(next_states).gather(
                    1, next_actions.unsqueeze(1)
                ).squeeze(1)
                target_next_q = torch.where(dones.bool(), 0.0, target_next_q)
                expected_values = rewards + self.config.gamma * target_next_q
            
            td_errors = (current_values - expected_values).abs()
            dqn_loss = (is_weights * F.smooth_l1_loss(current_values, expected_values, reduction='none')).mean()
            
            losses['dqn_loss'] = dqn_loss
        
        losses['td_errors'] = td_errors.detach()
        
        # === CQL Regularization ===
        if self.config.cql_alpha > 0:
            q_all = self.online_net.get_q_values(states)
            cql_reg = cql_loss(
                q_all,
                actions,
                expected_values if 'expected_values' in locals() else rewards + self.config.gamma * next_q,
                alpha=self.config.cql_alpha,
                temp=self.config.cql_temp,
                min_q_weight=self.config.cql_min_q_weight,
            )
            losses['cql_loss'] = cql_reg
        else:
            losses['cql_loss'] = torch.tensor(0.0, device=self.device)
        
        # === Auxiliary Losses ===
        aux_loss = torch.tensor(0.0, device=self.device)
        if hasattr(self.online_net, 'config') and self.online_net.config.num_aux_heads > 0:
            online_out = self.online_net.forward_all(states)
            
            # Reward prediction
            pred_reward = online_out["aux"][0]
            aux_loss += F.mse_loss(pred_reward, rewards)
            
            # Next state feature prediction
            target_next_features = next_states.mean(dim=1)
            pred_next_features = online_out["aux"][1]
            aux_loss += F.mse_loss(pred_next_features, target_next_features)
            
            # Uncertainty estimation
            pred_uncertainty = online_out["aux"][2]
            aux_loss += F.mse_loss(pred_uncertainty, td_errors.detach())
            
            aux_loss = aux_loss / 3.0
        
        losses['aux_loss'] = aux_loss
        
        # === Policy Distillation ===
        policy_loss = torch.tensor(0.0, device=self.device)
        policy_logits = self.online_net.get_policy_logits(states)
        with torch.no_grad():
            q_for_policy = self.online_net.get_q_values(states) / 0.1
            policy_targets = F.softmax(q_for_policy, dim=-1).argmax(dim=-1)
        policy_loss = F.cross_entropy(policy_logits, policy_targets)
        losses['policy_loss'] = policy_loss
        
        # === World Model Loss ===
        wm_loss = torch.tensor(0.0, device=self.device)
        if self.world_model is not None:
            # Prepare sequence (single step for now, can extend to sequences)
            obs_seq = states.unsqueeze(0)  # (1, batch, obs_dim)
            action_seq = F.one_hot(actions, self.config.action_dim).float().unsqueeze(0)
            reward_seq = rewards.unsqueeze(0)
            continue_seq = (~dones.bool()).float().unsqueeze(0)
            
            wm_out = self.world_model.observe(obs_seq, action_seq, reward_seq, continue_seq)
            wm_loss = wm_out['total_loss']
        
        losses['wm_loss'] = wm_loss
        
        # === Total Loss ===
        total_loss = (
            losses['dqn_loss'] +
            self.config.cql_alpha * losses['cql_loss'] +
            0.1 * losses['aux_loss'] +
            0.05 * losses['policy_loss'] +
            (0.5 if self.world_model is not None else 0.0) * losses['wm_loss']
        )
        losses['total_loss'] = total_loss
        
        return losses
    
    def imagination_train_step(self, policy: nn.Module, num_batches: int = 4) -> dict:
        """
        Model-based policy improvement using imagination rollouts.
        """
        if self.world_model is None:
            return {}
        
        # Sample initial states from replay
        if self.config.use_per:
            transitions, _, _ = self.replay.sample(self.config.imagination_batch)
        else:
            transitions = self._rng.sample(self._simple_replay, self.config.imagination_batch)
        
        states = torch.from_numpy(np.stack([t.state for t in transitions])).to(self.device)
        
        # Encode to world model state
        with torch.no_grad():
            init_state = self.world_model.initial_state(states.shape[0], self.device)
            # Single step encode
            z_embed, _, _ = self.world_model.rssm.encoder(states)
            embed_flat = z_embed.view(states.shape[0], -1)
            deter = init_state['deter']
            deter = self.world_model.rssm.recurrent(deter, init_state['stoch'], 
                torch.zeros(states.shape[0], self.config.action_dim, device=self.device))
            post_logits, post_probs = self.world_model.rssm.posterior(deter, embed_flat)
            stoch = post_probs.view(states.shape[0], -1)
            init_state = {'deter': deter, 'stoch': stoch}
        
        # Imagine trajectory
        traj = self.world_model.imagine(policy, init_state, self.config.imagination_horizon)
        
        # Policy loss: maximize imagined return
        seq_len = traj['rewards'].shape[0]
        batch_size = traj['rewards'].shape[1]
        
        # Compute returns
        returns = torch.zeros(batch_size, device=self.device)
        discount = 1.0
        for t in reversed(range(seq_len)):
            returns = traj['rewards'][t] + self.config.gamma * traj['continues'][t] * returns
        
        # Policy gradient (REINFORCE with baseline)
        policy_inputs = torch.cat([traj['deter'], traj['stoch']], dim=-1)  # (seq, batch, dim)
        action_logits = policy(policy_inputs.view(-1, policy_inputs.shape[-1]))
        action_logits = action_logits.view(seq_len, batch_size, -1)
        
        actions = traj['actions']  # (seq, batch, action_dim)
        action_indices = actions.argmax(dim=-1)
        
        log_probs = F.log_softmax(action_logits, dim=-1)
        selected_log_probs = log_probs[torch.arange(seq_len).unsqueeze(1), 
                                       torch.arange(batch_size).unsqueeze(0),
                                       action_indices]
        
        # Advantage = return - value baseline
        with torch.no_grad():
            values = self.online_net.get_value(
                policy_inputs.view(-1, policy_inputs.shape[-1])
            ).view(seq_len, batch_size)
            advantages = returns.unsqueeze(0) - values
        
        policy_loss = -(selected_log_probs * advantages.detach()).mean()
        
        # Entropy bonus
        entropy = -(F.softmax(action_logits, dim=-1) * log_probs).sum(-1).mean()
        
        total_loss = policy_loss - 0.01 * entropy
        
        # Backward
        self.optimizer.zero_grad()
        if self.use_amp:
            with autocast(dtype=self.amp_dtype):
                self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(policy.parameters(), self.config.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
        
        return {
            'imagination_policy_loss': policy_loss.item(),
            'imagination_entropy': entropy.item(),
            'imagination_return': returns.mean().item(),
        }
    
    def save(self, path: Path, metadata: Optional[dict] = None) -> None:
        """Save complete training state."""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        payload = {
            'config': self.config.__dict__,
            'online_state_dict': self.online_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'ema_state_dict': self.ema.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'environment_steps': self.environment_steps,
            'training_steps': self.training_steps,
            'optimizer_steps': self.optimizer_steps,
            'accumulation_step': self.accumulation_step,
            'rng_state': self._rng.getstate(),
            'torch_rng_state': self._torch_generator.get_state(),
            'metadata': metadata or {},
        }
        
        if self.world_model is not None:
            payload['world_model_state_dict'] = self.world_model.state_dict()
        
        if self.config.use_per:
            payload['replay'] = {
                'data': self.replay._data,
                'tree': self.replay._tree,
                'pos': self.replay._pos,
                'size': self.replay._size,
                'max_priority': self.replay._max_priority,
                'frame': self.replay._frame,
                'beta': self.replay._beta,
            }
        else:
            payload['replay'] = list(self._simple_replay)
        
        if self.scaler is not None:
            payload['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(payload, path)
    
    def load(self, path: Path, resume: bool = True) -> dict:
        """Load training state."""
        payload = torch.load(path, map_location=self.device, weights_only=False)
        
        self.online_net.load_state_dict(payload['online_state_dict'])
        self.target_net.load_state_dict(payload['target_state_dict'])
        self.ema.load_state_dict(payload['ema_state_dict'])
        
        if self.world_model is not None and 'world_model_state_dict' in payload:
            self.world_model.load_state_dict(payload['world_model_state_dict'])
        
        if resume:
            self.optimizer.load_state_dict(payload['optimizer_state_dict'])
            self.environment_steps = payload['environment_steps']
            self.training_steps = payload['training_steps']
            self.optimizer_steps = payload['optimizer_steps']
            self.accumulation_step = payload.get('accumulation_step', 0)
            self._rng.setstate(payload['rng_state'])
            self._torch_generator.set_state(payload['torch_rng_state'])
            
            if self.config.use_per and 'replay' in payload:
                rp = payload['replay']
                self.replay._data = rp['data']
                self.replay._tree = rp['tree']
                self.replay._pos = rp['pos']
                self.replay._size = rp['size']
                self.replay._max_priority = rp['max_priority']
                self.replay._frame = rp.get('frame', 0)
                self.replay._beta = rp.get('beta', self.config.per_beta_start)
            elif not self.config.use_per and 'replay' in payload:
                self._simple_replay = deque(payload['replay'], maxlen=self.config.replay_capacity)
            
            if self.scaler is not None and 'scaler_state_dict' in payload:
                self.scaler.load_state_dict(payload['scaler_state_dict'])
        
        return payload.get('metadata', {})


# ============================================================================
# Factory
# ============================================================================

def create_trainer(
    input_dim: int = 40,
    action_dim: int = 6,
    hidden_dim: int = 512,
    num_residual_blocks: int = 6,
    batch_size: int = 256,
    gradient_accumulation_steps: int = 2,
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    use_moe: bool = True,
    use_distributional: bool = True,
    use_world_model: bool = True,
    use_mixed_precision: bool = True,
    use_spectral_norm: bool = True,
    cql_alpha: float = 1.0,
    compile_model: bool = True,
    compile_mode: str = "reduce-overhead",
    warmup_steps: int = 1000,
    device: Optional[torch.device] = None,
    seed: int = 0,
) -> Trainer:
    """Factory for creating trainer with sensible defaults."""
    config = TrainerConfig(
        input_dim=input_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        num_residual_blocks=num_residual_blocks,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        gamma=gamma,
        use_moe=use_moe,
        use_distributional=use_distributional,
        use_world_model=use_world_model,
        use_mixed_precision=use_mixed_precision,
        use_spectral_norm=use_spectral_norm,
        cql_alpha=cql_alpha,
        compile_model=compile_model,
        compile_mode=compile_mode,
        warmup_steps=warmup_steps,
    )
    return Trainer(config, device=device, seed=seed)