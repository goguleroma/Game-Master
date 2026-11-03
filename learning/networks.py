"""
Modern neural network components for Game Master.

Architecture improvements:
- ResNet-style blocks with LayerNorm
- SwiGLU activation (better than ReLU for deeper nets)
- Mixture of Experts (MoE) for action head
- Spectral Normalization on all linear layers
- EMA (Exponential Moving Average) of weights
"""
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


# ============================================================================
# Activation Functions
# ============================================================================

class SwiGLU(nn.Module):
    """
    SwiGLU activation: x * SiLU(gate) = x * sigmoid(gate) * gate
    Better gradient flow than ReLU for deep networks.
    """
    def __init__(self, dim: int, hidden_mult: int = 4):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_mult * dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_mult * dim, bias=False)
        self.down_proj = nn.Linear(hidden_mult * dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(F.silu(gate) * up)


class SwiGLUBlock(nn.Module):
    """Single SwiGLU feedforward block with residual connection and LayerNorm."""
    def __init__(self, dim: int, hidden_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.swish = SwiGLU(dim, hidden_mult)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.swish(self.norm(x)))


# ============================================================================
# Spectral Normalization
# ============================================================================

def spectral_norm(module: nn.Module, name: str = 'weight', n_power_iterations: int = 1, eps: float = 1e-12) -> nn.Module:
    """
    Apply spectral normalization to a module's weight parameter.
    Constrains the spectral norm (largest singular value) to 1 for stability.
    """
    return nn.utils.spectral_norm(module, name=name, n_power_iterations=n_power_iterations, eps=eps)


def apply_spectral_norm(module: nn.Module, exclude: tuple[type, ...] = (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)) -> nn.Module:
    """Recursively apply spectral normalization to all Linear and Conv layers."""
    for name, child in module.named_children():
        if isinstance(child, (nn.Linear, nn.Conv1d, nn.Conv2d)) and not isinstance(child, exclude):
            setattr(module, name, spectral_norm(child))
        else:
            apply_spectral_norm(child, exclude)
    return module


# ============================================================================
# ResNet-style Blocks
# ============================================================================

class ResidualBlock(nn.Module):
    """
    Pre-activation ResNet block with LayerNorm and SwiGLU.
    Architecture: x -> LN -> SwiGLU -> Dropout -> Linear -> x + out
    """
    def __init__(self, dim: int, hidden_mult: int = 4, dropout: float = 0.1, use_spectral_norm: bool = True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.swish = SwiGLU(dim, hidden_mult)
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, dim, bias=False)
        if use_spectral_norm:
            self.linear = spectral_norm(self.linear)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = self.swish(x)
        x = self.dropout(x)
        x = self.norm2(x)
        x = self.linear(x)
        x = self.dropout2(x)
        return residual + x


class ResNetBackbone(nn.Module):
    """
    Deep ResNet backbone with configurable depth.
    Replaces the 3-layer MLP with N residual blocks.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        num_blocks: int = 6,
        hidden_mult: int = 4,
        dropout: float = 0.1,
        use_spectral_norm: bool = True,
        use_layer_scale: bool = True,
        init_scale: float = 1e-3,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim, bias=False)
        if use_spectral_norm:
            self.input_proj = spectral_norm(self.input_proj)
        
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, hidden_mult, dropout, use_spectral_norm)
            for _ in range(num_blocks)
        ])
        
        self.output_norm = nn.LayerNorm(hidden_dim)
        
        # LayerScale for stable training of very deep networks
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scales = nn.ParameterList([
                nn.Parameter(torch.full((hidden_dim,), init_scale))
                for _ in range(num_blocks)
            ])
        
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        for block in self.blocks:
            if hasattr(block, 'linear') and block.linear.weight is not None:
                nn.init.xavier_uniform_(block.linear.weight, gain=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for i, block in enumerate(self.blocks):
            residual = block(x)
            if self.use_layer_scale:
                residual = residual * self.layer_scales[i]
            x = x + residual
        return self.output_norm(x)


# ============================================================================
# Mixture of Experts (MoE) for Action Head
# ============================================================================

class MoELayer(nn.Module):
    """
    Mixture of Experts layer for action head.
    Increases model capacity without proportional parameter increase.
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_experts: int = 8,
        expert_capacity: int = 2,
        hidden_mult: int = 4,
        dropout: float = 0.1,
        use_spectral_norm: bool = True,
        router_noise: float = 0.1,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.expert_capacity = expert_capacity
        self.router_noise = router_noise
        
        # Router (gating network)
        self.router = nn.Linear(input_dim, num_experts, bias=False)
        if use_spectral_norm:
            self.router = spectral_norm(self.router)
        
        # Experts (each is a small MLP)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_mult * input_dim, bias=False),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_mult * input_dim, output_dim, bias=False),
            )
            for _ in range(num_experts)
        ])
        
        if use_spectral_norm:
            for expert in self.experts:
                expert[0] = spectral_norm(expert[0])
                expert[3] = spectral_norm(expert[3])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim)
        Returns:
            out: (batch, output_dim)
        """
        batch_size = x.shape[0]
        
        # Compute routing weights
        router_logits = self.router(x)  # (batch, num_experts)
        
        if self.training and self.router_noise > 0:
            # Add noise for exploration during training
            router_logits = router_logits + torch.randn_like(router_logits) * self.router_noise
        
        router_probs = F.softmax(router_logits, dim=-1)  # (batch, num_experts)
        
        # Top-k routing (differentiable via straight-through or soft)
        # Use soft routing for gradient flow, hard for inference
        if self.training:
            # Soft routing: weighted sum of expert outputs
            expert_outputs = []
            for expert in self.experts:
                expert_outputs.append(expert(x))
            expert_outputs = torch.stack(expert_outputs, dim=1)  # (batch, num_experts, output_dim)
            out = torch.einsum('be,beo->bo', router_probs, expert_outputs)
        else:
            # Hard routing: top-k experts
            topk_probs, topk_indices = router_probs.topk(self.expert_capacity, dim=-1)
            topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
            out = torch.zeros(batch_size, self.experts[0][-1].out_features, device=x.device, dtype=x.dtype)
            for k in range(self.expert_capacity):
                idx = topk_indices[:, k]
                prob = topk_probs[:, k].unsqueeze(-1)
                for e in range(self.num_experts):
                    mask = (idx == e)
                    if mask.any():
                        out[mask] += prob[mask] * self.experts[e](x[mask])
        
        return out


# ============================================================================
# EMA (Exponential Moving Average) for Model Weights
# ============================================================================

class EMAModel(nn.Module):
    """
    Exponential Moving Average of model weights for evaluation.
    Maintains a shadow copy of parameters updated with EMA decay.
    """
    def __init__(self, model: nn.Module, decay: float = 0.9999, update_after_step: int = 0, update_every: int = 1):
        super().__init__()
        self.model = model
        self.decay = decay
        self.update_after_step = update_after_step
        self.update_every = update_every
        self.step = 0
        
        # Create shadow parameters
        self.shadow_params = {}
        self.param_names = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow_params[name] = param.detach().clone()
                self.param_names.append(name)
        
        # Also track buffers (BatchNorm stats)
        self.shadow_buffers = {}
        for name, buffer in model.named_buffers():
            self.shadow_buffers[name] = buffer.detach().clone()

    @torch.no_grad()
    def update(self, model: Optional[nn.Module] = None) -> None:
        """Update EMA parameters."""
        if model is None:
            model = self.model
        
        self.step += 1
        if self.step < self.update_after_step or self.step % self.update_every != 0:
            return
        
        # Compute decay (can warm up)
        decay = min(self.decay, (1 + self.step) / (10 + self.step))
        
        for name in self.param_names:
            param = dict(model.named_parameters())[name]
            if param.requires_grad:
                self.shadow_params[name].lerp_(param.detach(), 1 - decay)
        
        for name in self.shadow_buffers:
            buffer = dict(model.named_buffers())[name]
            self.shadow_buffers[name].lerp_(buffer.detach(), 1 - decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        """Copy EMA parameters to model for evaluation."""
        for name in self.param_names:
            param = dict(model.named_parameters())[name]
            if param.requires_grad:
                param.copy_(self.shadow_params[name])
        
        for name in self.shadow_buffers:
            buffer = dict(model.named_buffers())[name]
            buffer.copy_(self.shadow_buffers[name])

    @torch.no_grad()
    def store(self, model: nn.Module) -> dict:
        """Store current model parameters and return them for later restoration."""
        stored = {}
        for name in self.param_names:
            param = dict(model.named_parameters())[name]
            if param.requires_grad:
                stored[name] = param.detach().clone()
        for name in self.shadow_buffers:
            buffer = dict(model.named_buffers())[name]
            stored[name] = buffer.detach().clone()
        return stored

    @torch.no_grad()
    def restore(self, model: nn.Module, stored: dict) -> None:
        """Restore model parameters from stored dict."""
        for name, tensor in stored.items():
            if name in self.param_names:
                param = dict(model.named_parameters())[name]
                if param.requires_grad:
                    param.copy_(tensor)
            elif name in self.shadow_buffers:
                buffer = dict(model.named_buffers())[name]
                buffer.copy_(tensor)

    def state_dict(self) -> dict:
        return {
            'shadow_params': {k: v.clone() for k, v in self.shadow_params.items()},
            'shadow_buffers': {k: v.clone() for k, v in self.shadow_buffers.items()},
            'step': self.step,
            'decay': self.decay,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.shadow_params = {k: v.clone() for k, v in state_dict['shadow_params'].items()}
        self.shadow_buffers = {k: v.clone() for k, v in state_dict['shadow_buffers'].items()}
        self.step = state_dict['step']
        self.decay = state_dict.get('decay', self.decay)


# ============================================================================
# Distributional RL: C51 (Categorical DQN)
# ============================================================================

class CategoricalHead(nn.Module):
    """
    C51 Categorical DQN head.
    Outputs a distribution over value atoms instead of scalar Q-values.
    """
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        num_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        hidden_dim: int = 256,
        use_spectral_norm: bool = True,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.register_buffer('supports', torch.linspace(v_min, v_max, num_atoms))
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, action_dim * num_atoms, bias=False),
        )
        if use_spectral_norm:
            self.net[0] = spectral_norm(self.net[0])
            self.net[-1] = spectral_norm(self.net[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim)
        Returns:
            logits: (batch, action_dim, num_atoms)
        """
        logits = self.net(x)
        return logits.view(-1, self.action_dim, self.num_atoms)

    def get_q_values(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute expected Q-values from distribution."""
        probs = F.softmax(logits, dim=-1)
        return torch.sum(probs * self.supports, dim=-1)

    def get_distribution(self, logits: torch.Tensor) -> torch.Tensor:
        """Get probability distribution."""
        return F.softmax(logits, dim=-1)


# ============================================================================
# Modern Q-Network (replaces _QNetwork)
# ============================================================================

@dataclass
class ModernQNetworkConfig:
    input_dim: int
    action_dim: int
    hidden_dim: int = 512
    num_residual_blocks: int = 6
    hidden_mult: int = 4
    dropout: float = 0.1
    use_spectral_norm: bool = True
    use_layer_scale: bool = True
    # MoE settings
    use_moe: bool = True
    num_experts: int = 8
    expert_capacity: int = 2
    # Distributional RL
    use_distributional: bool = True
    num_atoms: int = 51
    v_min: float = -10.0
    v_max: float = 10.0
    # Auxiliary heads
    num_aux_heads: int = 3
    aux_hidden_dim: int = 128


class ModernQNetwork(nn.Module):
    """
    Modern Q-Network with:
    - ResNet backbone with LayerNorm + SwiGLU
    - Spectral Normalization
    - Mixture of Experts action head
    - C51 Distributional RL head
    - Multiple auxiliary heads
    - EMA-ready
    """
    def __init__(self, config: ModernQNetworkConfig):
        super().__init__()
        self.config = config
        
        # Backbone
        self.backbone = ResNetBackbone(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            num_blocks=config.num_residual_blocks,
            hidden_mult=config.hidden_mult,
            dropout=config.dropout,
            use_spectral_norm=config.use_spectral_norm,
            use_layer_scale=config.use_layer_scale,
        )
        
        # Main head: Distributional (C51) or standard Q
        if config.use_distributional:
            self.q_head = CategoricalHead(
                input_dim=config.hidden_dim,
                action_dim=config.action_dim,
                num_atoms=config.num_atoms,
                v_min=config.v_min,
                v_max=config.v_max,
                hidden_dim=config.hidden_dim // 2,
                use_spectral_norm=config.use_spectral_norm,
            )
        else:
            # Standard Q head with MoE
            if config.use_moe:
                self.q_head = MoELayer(
                    input_dim=config.hidden_dim,
                    output_dim=config.action_dim,
                    num_experts=config.num_experts,
                    expert_capacity=config.expert_capacity,
                    hidden_mult=config.hidden_mult,
                    dropout=config.dropout,
                    use_spectral_norm=config.use_spectral_norm,
                )
            else:
                self.q_head = nn.Sequential(
                    nn.Linear(config.hidden_dim, config.hidden_dim // 2, bias=False),
                    nn.LayerNorm(config.hidden_dim // 2),
                    nn.SiLU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.hidden_dim // 2, config.action_dim, bias=False),
                )
                if config.use_spectral_norm:
                    self.q_head[0] = spectral_norm(self.q_head[0])
                    self.q_head[-1] = spectral_norm(self.q_head[-1])
        
        # Value head (for advantage decomposition)
        self.value_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 4, bias=False),
            nn.LayerNorm(config.hidden_dim // 4),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 4, 1, bias=False),
        )
        if config.use_spectral_norm:
            self.value_head[0] = spectral_norm(self.value_head[0])
            self.value_head[-1] = spectral_norm(self.value_head[-1])
        
        # Policy head (for behavior cloning / distillation)
        self.policy_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2, bias=False),
            nn.LayerNorm(config.hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.action_dim, bias=False),
        )
        if config.use_spectral_norm:
            self.policy_head[0] = spectral_norm(self.policy_head[0])
            self.policy_head[-1] = spectral_norm(self.policy_head[-1])
        
        # Auxiliary heads
        self.aux_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_dim, config.aux_hidden_dim, bias=False),
                nn.LayerNorm(config.aux_hidden_dim),
                nn.SiLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.aux_hidden_dim, 1, bias=False),
            )
            for _ in range(config.num_aux_heads)
        ])
        if config.use_spectral_norm:
            for head in self.aux_heads:
                head[0] = spectral_norm(head[0])
                head[-1] = spectral_norm(head[-1])

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Standard forward for compatibility - returns Q-values."""
        return self.get_q_values(state)

    def get_q_values(self, state: torch.Tensor) -> torch.Tensor:
        """Get Q-values (expected value for distributional)."""
        features = self.backbone(state)
        if self.config.use_distributional:
            logits = self.q_head(features)
            return self.q_head.get_q_values(logits)
        else:
            return self.q_head(features)

    def get_q_distribution(self, state: torch.Tensor) -> torch.Tensor:
        """Get full Q-distribution (C51)."""
        if not self.config.use_distributional:
            raise ValueError("Distributional head not enabled")
        features = self.backbone(state)
        logits = self.q_head(features)
        return self.q_head.get_distribution(logits)

    def get_q_logits(self, state: torch.Tensor) -> torch.Tensor:
        """Get raw logits for C51 loss computation."""
        if not self.config.use_distributional:
            raise ValueError("Distributional head not enabled")
        features = self.backbone(state)
        return self.q_head(features)

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """Get state value V(s)."""
        features = self.backbone(state)
        return self.value_head(features).squeeze(-1)

    def get_advantage(self, state: torch.Tensor) -> torch.Tensor:
        """Get advantage A(s,a) = Q(s,a) - V(s)."""
        q = self.get_q_values(state)
        v = self.get_value(state).unsqueeze(-1)
        return q - v

    def get_policy_logits(self, state: torch.Tensor) -> torch.Tensor:
        """Get policy logits for behavior cloning."""
        features = self.backbone(state)
        return self.policy_head(features)

    def get_aux_predictions(self, state: torch.Tensor) -> list[torch.Tensor]:
        """Get auxiliary task predictions."""
        features = self.backbone(state)
        return [head(features).squeeze(-1) for head in self.aux_heads]

    def forward_all(self, state: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass returning all heads."""
        features = self.backbone(state)
        out = {
            "features": features,
            "value": self.value_head(features).squeeze(-1),
            "policy_logits": self.policy_head(features),
            "aux": [head(features).squeeze(-1) for head in self.aux_heads],
        }
        if self.config.use_distributional:
            logits = self.q_head(features)
            out["q_logits"] = logits
            out["q_values"] = self.q_head.get_q_values(logits)
            out["q_dist"] = self.q_head.get_distribution(logits)
        else:
            out["q_values"] = self.q_head(features)
        return out


# ============================================================================
# Factory Functions
# ============================================================================

def create_modern_q_network(
    input_dim: int,
    action_dim: int,
    hidden_dim: int = 512,
    num_residual_blocks: int = 6,
    use_moe: bool = True,
    use_distributional: bool = True,
    num_atoms: int = 51,
    use_spectral_norm: bool = True,
    dropout: float = 0.1,
) -> ModernQNetwork:
    """Factory for creating a modern Q-network with sensible defaults."""
    config = ModernQNetworkConfig(
        input_dim=input_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        num_residual_blocks=num_residual_blocks,
        use_moe=use_moe,
        use_distributional=use_distributional,
        num_atoms=num_atoms,
        use_spectral_norm=use_spectral_norm,
        dropout=dropout,
    )
    return ModernQNetwork(config)


# ============================================================================
# Utility: C51 Projection (for distributional Bellman update)
# ============================================================================

def categorical_projection(
    next_dist: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    supports: torch.Tensor,
) -> torch.Tensor:
    """
    Project next-state distribution to current state (C51 algorithm).
    
    Args:
        next_dist: (batch, action_dim, num_atoms) - target distribution
        rewards: (batch,) - rewards
        dones: (batch,) - done flags
        gamma: discount factor
        supports: (num_atoms,) - value atoms
    Returns:
        projected_dist: (batch, action_dim, num_atoms)
    """
    batch_size, action_dim, num_atoms = next_dist.shape
    v_min, v_max = supports[0].item(), supports[-1].item()
    delta_z = (v_max - v_min) / (num_atoms - 1)
    
    # Compute target supports: r + γ * z
    # rewards: (batch,) -> (batch, 1, 1)
    # supports: (num_atoms,) -> (1, 1, num_atoms)
    target_supports = rewards.view(-1, 1, 1) + gamma * supports.view(1, 1, -1) * (~dones).view(-1, 1, 1).float()
    target_supports = target_supports.clamp(v_min, v_max)
    
    # Project onto fixed supports
    # b = (target_supports - v_min) / delta_z
    b = (target_supports - v_min) / delta_z
    l = b.floor().long().clamp(0, num_atoms - 1)
    u = b.ceil().long().clamp(0, num_atoms - 1)
    
    # Distribute probability mass
    projected = torch.zeros_like(next_dist)
    offset = torch.arange(batch_size, device=next_dist.device).view(-1, 1, 1) * action_dim * num_atoms
    offset = offset + torch.arange(action_dim, device=next_dist.device).view(1, -1, 1) * num_atoms
    
    # Flatten for scatter_add
    next_dist_flat = next_dist.view(-1)
    l_flat = (l + offset).view(-1)
    u_flat = (u + offset).view(-1)
    
    # Weight for lower and upper
    w_l = (u.float() - b).view(-1)
    w_u = (b - l.float()).view(-1)
    
    projected_flat = projected.view(-1)
    projected_flat.index_add_(0, l_flat, next_dist_flat * w_l)
    projected_flat.index_add_(0, u_flat, next_dist_flat * w_u)
    
    return projected.view(batch_size, action_dim, num_atoms)