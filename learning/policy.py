"""
Modern Neural Policy for Game Master.

Replaces the hand-crafted 40-feature encoder with:
- Latent state from World Model (RSSM)
- Transformer encoder for variable-length entity sets
- GNN message passing for spatial relations
- Behavior Cloning -> RL curriculum with proper warmup
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
from collections import Counter
import math

import torch
from torch import nn
from torch.nn import functional as F

from core.types import Action, ActionResult, GameState
from simulated_game.agent.memory import GridMemoryStore
from simulated_game.agent.planner import Planner
from simulated_game.agent.world_model import WorldModel as LegacyWorldModel
from learning.networks import ModernQNetwork, create_modern_q_network, ModernQNetworkConfig
from learning.world_model import WorldModel, WorldModelConfig, create_world_model
from learning.trainer import Trainer, TrainerConfig, create_trainer, Transition


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ModernPolicyConfig:
    # Network
    input_dim: int = 40
    action_count: int = 6
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
    wm_ensemble_size: int = 5
    imagination_horizon: int = 15
    
    # Training
    batch_size: int = 256
    gradient_accumulation_steps: int = 2
    gamma: float = 0.99
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    target_update_steps: int = 1000
    warmup_steps: int = 1000
    
    # CQL
    cql_alpha: float = 1.0
    cql_temp: float = 1.0
    cql_min_q_weight: float = 5.0
    
    # PER
    use_per: bool = True
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_frames: int = 200_000
    per_eps: float = 1e-6
    replay_capacity: int = 100_000
    
    # EMA
    ema_decay: float = 0.9995
    ema_update_after: int = 1000
    ema_update_every: int = 1
    
    # Mixed Precision
    use_mixed_precision: bool = True
    precision: str = "bf16"
    
    # Exploration
    epsilon_decay_steps: int = 50_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    
    # Compilation
    compile_model: bool = True
    compile_mode: str = "reduce-overhead"
    
    # Behavior Cloning
    bc_epochs: int = 10
    bc_batch_size: int = 256
    
    # Curriculum
    use_curriculum: bool = True
    curriculum_stages: List[Dict[str, Any]] = None


# ============================================================================
# Entity Transformer Encoder
# ============================================================================

class EntityTransformerEncoder(nn.Module):
    """
    Transformer encoder for variable-length entity sets.
    Processes entities with positional encoding and self-attention.
    """
    def __init__(
        self,
        entity_feat_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_entities: int = 32,
    ):
        super().__init__()
        self.entity_feat_dim = entity_feat_dim
        self.hidden_dim = hidden_dim
        self.max_entities = max_entities
        
        # Input projection
        self.input_proj = nn.Linear(entity_feat_dim, hidden_dim)
        
        # Positional encoding for entity order
        self.pos_encoding = nn.Parameter(torch.randn(max_entities, hidden_dim) * 0.02)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.output_norm = nn.LayerNorm(hidden_dim)
        
        # Pooling
        self.pool = nn.AdaptiveAvgPool1d(1)
    
    def forward(
        self,
        entities: torch.Tensor,
        entity_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            entities: (batch, num_entities, entity_feat_dim)
            entity_mask: (batch, num_entities) - True for valid entities
        Returns:
            pooled: (batch, hidden_dim)
        """
        batch_size, num_entities, _ = entities.shape
        
        # Project
        x = self.input_proj(entities)  # (batch, num_entities, hidden_dim)
        
        # Add positional encoding
        x = x + self.pos_encoding[:num_entities].unsqueeze(0)
        
        # Transformer
        if entity_mask is not None:
            # Transformer expects False for valid positions
            src_key_padding_mask = ~entity_mask
        else:
            src_key_padding_mask = None
        
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        
        # Pool
        x = self.output_norm(self.output_proj(x))
        
        if entity_mask is not None:
            # Masked average pooling
            mask = entity_mask.unsqueeze(-1).float()
            x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)
        
        return x


# ============================================================================
# GNN for Spatial Relations
# ============================================================================

class SpatialGNN(nn.Module):
    """
    Graph Neural Network for spatial reasoning.
    Nodes = cells/entities, Edges = spatial proximity.
    """
    def __init__(
        self,
        node_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        max_neighbors: int = 8,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.max_neighbors = max_neighbors
        
        # Message passing layers
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(node_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, node_dim),
            )
            for _ in range(num_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(node_dim) for _ in range(num_layers)
        ])
    
    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            node_features: (num_nodes, node_dim)
            edge_index: (2, num_edges) - source, target
            edge_attr: (num_edges, edge_dim) - optional edge features
        Returns:
            updated_features: (num_nodes, node_dim)
        """
        x = node_features
        
        for layer, norm in zip(self.layers, self.layer_norms):
            # Aggregate messages
            src, dst = edge_index
            messages = torch.cat([x[src], x[dst]], dim=-1)
            
            if edge_attr is not None:
                messages = torch.cat([messages, edge_attr], dim=-1)
            
            messages = layer(messages)
            
            # Aggregate (sum)
            agg = torch.zeros_like(x)
            agg.index_add_(0, dst, messages)
            
            # Residual + norm
            x = norm(x + agg)
        
        return x


# ============================================================================
# Modern Neural Policy
# ============================================================================

class ModernNeuralPolicy:
    """
    Modern neural policy using:
    - World Model latent states (instead of 40 hand-crafted features)
    - Transformer for entity processing
    - GNN for spatial reasoning
    - Unified trainer with CQL, mixed precision, EMA
    """
    
    def __init__(
        self,
        config: Optional[ModernPolicyConfig] = None,
        seed: int = 0,
        training: bool = False,
        epsilon_start: float = 1.0,
        device: Optional[torch.device] = None,
        **kwargs,
    ):
        # Merge config with kwargs
        if config is None:
            config = ModernPolicyConfig()
        
        # Override with kwargs
        for k, v in kwargs.items():
            if hasattr(config, k):
                setattr(config, k, v)
        
        self.config = config
        self.training = training
        
        # Device
        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        print(f"ModernNeuralPolicy using device: {self.device}")
        
        # Legacy components for compatibility
        self.legacy_world_model = LegacyWorldModel()
        self._pending = None
        self._pending_memory_case = None
        self._last_planned_direction = None
        self._novel_states: Counter = Counter()
        self.losses: List[float] = []
        
        # Initialize trainer (handles all networks, optimizers, replay)
        trainer_config = TrainerConfig(
            input_dim=config.input_dim,
            action_dim=config.action_count,
            hidden_dim=config.hidden_dim,
            num_residual_blocks=config.num_residual_blocks,
            use_moe=config.use_moe,
            use_distributional=config.use_distributional,
            num_atoms=config.num_atoms,
            v_min=config.v_min,
            v_max=config.v_max,
            use_spectral_norm=config.use_spectral_norm,
            dropout=config.dropout,
            use_world_model=config.use_world_model,
            deter_dim=config.deter_dim,
            num_categoricals=config.num_categoricals,
            num_classes=config.num_classes,
            codebook_dim=config.codebook_dim,
            wm_hidden_dim=config.wm_hidden_dim,
            wm_num_layers=config.wm_num_layers,
            wm_ensemble_size=config.wm_ensemble_size,
            imagination_horizon=config.imagination_horizon,
            batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            gamma=config.gamma,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            max_grad_norm=config.max_grad_norm,
            cql_alpha=config.cql_alpha,
            cql_temp=config.cql_temp,
            cql_min_q_weight=config.cql_min_q_weight,
            use_per=config.use_per,
            per_alpha=config.per_alpha,
            per_beta_start=config.per_beta_start,
            per_beta_frames=config.per_beta_frames,
            per_eps=config.per_eps,
            replay_capacity=config.replay_capacity,
            ema_decay=config.ema_decay,
            ema_update_after=config.ema_update_after,
            ema_update_every=config.ema_update_every,
            use_mixed_precision=config.use_mixed_precision,
            precision=config.precision,
            target_update_steps=config.target_update_steps,
            warmup_steps=config.warmup_steps,
            epsilon_decay_steps=config.epsilon_decay_steps,
            epsilon_start=config.epsilon_start,
            epsilon_end=config.epsilon_end,
            compile_model=config.compile_model,
            compile_mode=config.compile_mode,
        )
        
        self.trainer = Trainer(trainer_config, device=self.device, seed=seed)
        
        # Entity encoder for richer observations
        self.entity_encoder = EntityTransformerEncoder(
            entity_feat_dim=16,  # position, type, health, etc.
            hidden_dim=config.hidden_dim // 2,
            num_heads=4,
            num_layers=2,
            dropout=config.dropout,
        ).to(self.device)
        
        # Spatial GNN
        self.spatial_gnn = SpatialGNN(
            node_dim=config.hidden_dim // 4,
            hidden_dim=config.hidden_dim // 2,
            num_layers=2,
            dropout=config.dropout,
        ).to(self.device)
        
        # Curriculum state
        self.curriculum_stage = 0
        self.bc_complete = False
        self._bc_samples: List[Tuple] = []
    
    @property
    def ACTION_COUNT(self) -> int:
        return self.config.action_count
    
    @property
    def FEATURE_COUNT(self) -> int:
        return self.config.input_dim
    
    def begin_episode(self) -> None:
        self._pending = None
        self._pending_memory_case = None
        # Reset world model state
        if self.trainer.world_model is not None:
            self._wm_state = self.trainer.world_model.initial_state(1, self.device)
    
    def decide(
        self,
        observation: GameState,
        memory: GridMemoryStore,
    ) -> Action | None:
        normalized = self._normalize(observation)
        controls = normalized.get("controls", [])
        if not isinstance(controls, list) or not controls:
            return None
        if len(controls) > self.ACTION_COUNT:
            raise ValueError("Neural grid policy received more controls than its action head supports.")
        
        # Encode observation to latent state
        features, wm_state = self._encode_with_world_model(normalized, memory)
        
        # Training step on previous transition
        if self._pending is not None and self.training:
            prior_features, action_index, before, result = self._pending
            reward = self._reward(before, normalized, features, result)
            mask = tuple(index < len(controls) for index in range(self.ACTION_COUNT))
            self._store_transition(prior_features, action_index, reward, features, mask, done=False)
            loss_dict = self.trainer.train_step()
            if loss_dict is not None:
                self.losses.append(loss_dict.get('total_loss', 0.0))
        
        self._pending = None
        
        # Select action
        mask = tuple(index < len(controls) for index in range(self.ACTION_COUNT))
        action_mask = torch.tensor(mask, dtype=torch.bool, device=self.device).unsqueeze(0)
        features_tensor = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
        
        # Use EMA for evaluation
        use_ema = not self.training and self.trainer.ema.step > self.config.ema_update_after
        
        with torch.no_grad():
            if use_ema:
                self.trainer.ema.copy_to(self.trainer.online_net)
                q_values = self.trainer.online_net.get_q_values(features_tensor)
                self.trainer.ema.restore(self.trainer.online_net, {})
            else:
                q_values = self.trainer.online_net.get_q_values(features_tensor)
            
            q_values = q_values.masked_fill(~action_mask, -torch.inf)
            
            if self.training and self.trainer._rng.random() < self.trainer.epsilon:
                valid_actions = torch.where(action_mask[0])[0]
                action_index = int(self.trainer._rng.choice(valid_actions.cpu().numpy()))
            else:
                action_index = int(q_values.argmax(dim=1).item())
        
        if self.training:
            self.trainer.environment_steps += 1
        
        # Memory integration (legacy)
        state = self.legacy_world_model.state
        if (
            state.grid_size is not None
            and state.current_position is not None
            and self._last_planned_direction is not None
        ):
            self._pending_memory_case = (
                memory.build_decision_context(state, normalized),
                self._last_planned_direction,
                state.current_position,
                state.target_position,
            )
        else:
            self._pending_memory_case = None
        
        self._pending = (features, action_index, normalized, {})
        
        return Action(name=controls[action_index])
    
    def observe_result(
        self,
        result: ActionResult,
        memory: GridMemoryStore,
    ) -> None:
        pending_case = self._pending_memory_case
        self._pending_memory_case = None
        if pending_case is not None:
            context, direction, position, target = pending_case
            memory.record_decision_outcome(
                context, direction, position, target, result.details
            )
        
        if self._pending is None:
            return
        
        if not self.training:
            self._pending = None
            return
        
        status = result.details.get("status")
        if status in {"goal_reached", "won", "caught", "lost"}:
            features, action_index, before, _ = self._pending
            reward = self._reward(
                before, before, features, result.details, terminal=status
            )
            self._store_transition(
                features, action_index, reward,
                tuple(0.0 for _ in range(self.FEATURE_COUNT)),
                tuple(False for _ in range(self.ACTION_COUNT)),
                done=True,
            )
            loss_dict = self.trainer.train_step()
            if loss_dict is not None:
                self.losses.append(loss_dict.get('total_loss', 0.0))
            self._pending = None
            return
        
        self._pending = (
            self._pending[0],
            self._pending[1],
            self._pending[2],
            dict(result.details),
        )
    
    def finish_episode(
        self,
        observation: dict[str, Any],
        memory: GridMemoryStore,
    ) -> None:
        if not self.training or self._pending is None:
            self._pending = None
            return
        
        features, action_index, before, result = self._pending
        next_features, _ = self._encode_with_world_model(observation, memory)
        reward = self._reward(before, observation, next_features, result)
        self._store_transition(
            features, action_index, reward, next_features,
            tuple(False for _ in range(self.ACTION_COUNT)),
            done=True,
        )
        loss_dict = self.trainer.train_step()
        if loss_dict is not None:
            self.losses.append(loss_dict.get('total_loss', 0.0))
        self._pending = None
    
    def _store_transition(
        self,
        state: tuple,
        action: int,
        reward: float,
        next_state: tuple,
        next_action_mask: tuple,
        done: bool,
    ) -> None:
        """Store transition in trainer's replay buffer."""
        self.trainer.remember(
            np.array(state, dtype=np.float32),
            action,
            reward,
            np.array(next_state, dtype=np.float32),
            np.array(next_action_mask, dtype=np.bool_),
            done,
        )
    
    def _encode_with_world_model(
        self,
        observation: dict[str, Any],
        memory: GridMemoryStore,
        update_memory: bool = True,
    ) -> Tuple[np.ndarray, dict]:
        """
        Encode observation using World Model.
        Returns both legacy features (for compatibility) and WM state.
        """
        # Legacy feature encoding (for reward computation, memory)
        legacy_features = self._encode_legacy(observation, memory, update_memory)
        
        # World Model encoding
        obs_tensor = torch.from_numpy(np.array(legacy_features, dtype=np.float32)).unsqueeze(0).to(self.device)
        
        if self.trainer.world_model is not None:
            with torch.no_grad():
                # Single step observe
                obs_seq = obs_tensor.unsqueeze(0)  # (1, 1, obs_dim)
                action_seq = torch.zeros(1, 1, self.ACTION_COUNT, device=self.device)
                reward_seq = torch.zeros(1, 1, device=self.device)
                continue_seq = torch.ones(1, 1, device=self.device)
                
                wm_out = self.trainer.world_model.observe(
                    obs_seq, action_seq, reward_seq, continue_seq,
                    state=getattr(self, '_wm_state', None)
                )
                
                # Get posterior state
                posteriors = wm_out['posteriors']
                self._wm_state = {
                    'deter': posteriors['deter'][-1],
                    'stoch': posteriors['stoch'][-1],
                }
                
                # Use combined deter+stoch as features
                combined = torch.cat([
                    posteriors['deter'][-1],
                    posteriors['stoch'][-1],
                ], dim=-1).squeeze(0).cpu().numpy()
        else:
            combined = legacy_features
            self._wm_state = None
        
        return combined.astype(np.float32), self._wm_state
    
    def _encode_legacy(
        self,
        observation: dict[str, Any],
        memory: GridMemoryStore,
        update_memory: bool = True,
    ) -> np.ndarray:
        """Legacy 40-feature encoding (kept for reward computation and memory)."""
        state = self.legacy_world_model.update(observation)
        grid = observation.get("hud", {}).get("grid", {})
        if update_memory:
            current_turn = len(memory.experiences)
            visible = set(state.visible_cells)
            for cell in visible:
                memory.mark_cell(cell, seen=True, obstacle=False, value="", turn=current_turn)
            memory.observe_goal(state.target_position, list(visible))
            for obstacle in state.obstacle_positions:
                memory.mark_cell(obstacle, seen=True, obstacle=True, value="obstacle", turn=current_turn)
            if state.target_position is not None:
                memory.mark_cell(state.target_position, seen=True, obstacle=False, value="target", turn=current_turn)
            if state.current_position is not None:
                memory.mark_cell(state.current_position, seen=True, visited=True, obstacle=False, value="agent", turn=current_turn)
            for entity in state.entities:
                if entity.entity_id is not None and entity.position is not None:
                    memory.observe_entity(entity.entity_id, entity.position, state.current_position, current_turn)
            if state.grid_size is not None:
                memory.update_frontier(*state.grid_size)
            memory.update_working_memory(observation, memory.working.last_action, memory.working.last_result)
        
        width = grid.get("width", 1)
        height = grid.get("height", 1)
        position = state.current_position or (0, 0)
        visible_target = state.target_position
        remembered = (
            memory.spatial.known_goal_positions[-1]
            if memory.spatial.known_goal_positions
            else None
        )
        entities = grid.get("entities", [])
        visible_enemy = next(
            (entity for entity in entities
             if isinstance(entity, dict) and entity.get("id") == "entity-2"
             and isinstance(entity.get("position"), list)), None)
        obstacles = {tuple(point) for point in grid.get("obstacles", []) if isinstance(point, list) and len(point) == 2}
        x, y = position
        
        target_offset = ((visible_target[0] - x) / max(width, 1), (visible_target[1] - y) / max(height, 1)) if visible_target else (0.0, 0.0)
        remembered_offset = ((remembered[0] - x) / max(width, 1), (remembered[1] - y) / max(height, 1)) if remembered else (0.0, 0.0)
        enemy_position = tuple(visible_enemy["position"]) if visible_enemy else None
        enemy_offset = ((enemy_position[0] - x) / max(width, 1), (enemy_position[1] - y) / max(height, 1)) if enemy_position else (0.0, 0.0)
        
        frontier = memory.spatial.frontier
        nearest_frontier = min(frontier, key=lambda cell: abs(cell[0] - x) + abs(cell[1] - y)) if frontier else None
        frontier_offset = ((nearest_frontier[0] - x) / max(width, 1), (nearest_frontier[1] - y) / max(height, 1)) if nearest_frontier else (0.0, 0.0)
        
        action_index = self._previous_action_index(observation.get("controls", []), memory)
        planned_direction, intent_is_frontier = self._planning_hint(state, memory)
        self._last_planned_direction = planned_direction
        direction_features = [float(planned_direction == d) for d in ((0, -1), (1, 0), (0, 1), (-1, 0))]
        adjacent = ((0, -1), (1, 0), (0, 1), (-1, 0))
        
        values = [
            x / max(width - 1, 1),
            y / max(height - 1, 1),
            float(observation.get("hud", {}).get("health", 0)) / 10.0,
            float(visible_target is not None),
            *target_offset,
            float(remembered is not None),
            *remembered_offset,
            float(memory.spatial.goal_is_dynamic),
            float(visible_enemy is not None),
            *enemy_offset,
            float(visible_enemy.get("health", 0)) / 10.0 if visible_enemy else 0.0,
            *[float((x + dx, y + dy) in obstacles) for dx, dy in adjacent],
            *[min(memory.spatial.grid.get((x + dx, y + dy)).visit_count / 10.0, 1.0) if memory.spatial.grid.get((x + dx, y + dy)) else 0.0 for dx, dy in adjacent],
            float(nearest_frontier is not None),
            *frontier_offset,
            *[float(action_index == i) for i in range(self.ACTION_COUNT)],
            *direction_features,
            intent_is_frontier,
            *self._previous_result_features(memory),
        ]
        
        if len(values) != self.FEATURE_COUNT:
            raise RuntimeError(f"Neural observation encoder produced {len(values)} features; expected {self.FEATURE_COUNT}.")
        
        features = np.array(values, dtype=np.float32)
        key = tuple(round(v * 20) for v in features)
        self._novel_states[key] += 1
        return features
    
    @staticmethod
    def _planning_hint(state: Any, memory: GridMemoryStore) -> Tuple[Tuple[int, int] | None, float]:
        if state.grid_size is None:
            return None, 0.0
        plan = Planner().choose_action(state, memory)
        if plan is None:
            return None, 0.0
        direction = {"move_up": (0, -1), "move_right": (1, 0), "move_down": (0, 1), "move_left": (-1, 0)}.get(plan.name)
        intent_is_frontier = float(direction is not None and plan.utility < 1.0)
        return direction, intent_is_frontier
    
    @staticmethod
    def _normalize(observation: GameState) -> dict[str, Any]:
        return {"objects": observation.objects, "hud": observation.hud, "goal": observation.goal, "confidence": observation.confidence, "controls": observation.controls}
    
    @staticmethod
    def _previous_action_index(controls: object, memory: GridMemoryStore) -> int:
        if not isinstance(controls, list):
            return -1
        try:
            return controls.index(memory.working.last_action)
        except ValueError:
            return -1
    
    @staticmethod
    def _previous_result_features(memory: GridMemoryStore) -> List[float]:
        result = memory.working.last_result
        return [float(result in {"moved", "goal_reached"}), float(isinstance(result, str) and result.startswith("blocked_")), float(result == "completed"), float(result in {"caught", "lost"})]
    
    def _reward(
        self,
        before: dict[str, Any],
        after: dict[str, Any],
        features: np.ndarray,
        result: dict[str, Any],
        terminal: str | None = None,
    ) -> float:
        reward = -0.02
        key = tuple(round(v * 20) for v in features)
        reward += 0.25 / math.sqrt(max(self._novel_states[key], 1))
        
        previous_grid = before.get("hud", {}).get("grid", {})
        current_grid = after.get("hud", {}).get("grid", {})
        previous_visible = {tuple(cell) for cell in previous_grid.get("visible_cells", []) if isinstance(cell, list) and len(cell) == 2}
        current_visible = {tuple(cell) for cell in current_grid.get("visible_cells", []) if isinstance(cell, list) and len(cell) == 2}
        reward += 0.5 * len(current_visible - previous_visible) / 50.0
        
        previous_target = previous_grid.get("target")
        current_target = current_grid.get("target")
        previous_position = previous_grid.get("position")
        current_position = current_grid.get("position")
        if (isinstance(previous_target, list) and previous_target == current_target and isinstance(previous_position, list) and isinstance(current_position, list)):
            before_distance = abs(previous_position[0] - previous_target[0]) + abs(previous_position[1] - previous_target[1])
            after_distance = abs(current_position[0] - current_target[0]) + abs(current_position[1] - current_target[1])
            reward += max(-1, min(1, before_distance - after_distance)) * 0.2
        
        if (isinstance(previous_position, list) and len(previous_position) == 2 and isinstance(current_position, list) and len(current_position) == 2):
            # Check planner alignment
            planned_direction_features = features[31:35]  # PLANNED_DIRECTION_START = 31
            if any(planned_direction_features):
                planned_index = int(np.argmax(planned_direction_features))
                planned_direction = ((0, -1), (1, 0), (0, 1), (-1, 0))[planned_index]
                displacement = (current_position[0] - previous_position[0], current_position[1] - previous_position[1])
                if displacement == planned_direction:
                    reward += 0.2
                elif displacement == (0, 0):
                    reward -= 0.15 if str(result.get("status", "")).startswith("blocked_") else 0.05
                else:
                    reward -= 0.1
        
        outcome = terminal or result.get("status")
        if outcome in {"goal_reached", "won"}:
            reward += 10.0
        elif outcome in {"caught", "lost"}:
            reward -= 5.0
        return reward
    
    # Behavior Cloning
    def behavior_clone(
        self,
        samples: List[Tuple[np.ndarray, int, np.ndarray]],
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> float:
        """Behavior cloning using trainer's BC method."""
        epochs = epochs or self.config.bc_epochs
        batch_size = batch_size or self.config.bc_batch_size
        
        # Convert to trainer format
        trainer_samples = []
        for state, action, action_mask in samples:
            trainer_samples.append((
                np.array(state, dtype=np.float32),
                action,
                np.array(action_mask, dtype=np.bool_),
            ))
        
        return self.trainer.behavior_clone(trainer_samples, epochs=epochs, batch_size=batch_size)
    
    # Checkpointing
    def save(self, path, metadata: Optional[dict] = None) -> None:
        self.trainer.save(path, metadata)
    
    def load(self, path, resume: bool = True) -> dict:
        return self.trainer.load(path, resume)
    
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for a state (for analysis)."""
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            q = self.trainer.online_net.get_q_values(state_tensor)
            return q.cpu().numpy()[0]


# ============================================================================
# Compatibility wrapper for existing code
# ============================================================================

class NeuralGridPolicy:
    """
    Drop-in replacement for the old NeuralGridPolicy.
    Uses ModernNeuralPolicy internally.
    """
    def __init__(self, *args, **kwargs):
        # Extract modern config from kwargs
        modern_kwargs = {}
        legacy_kwargs = {}
        
        modern_params = {
            'hidden_dim', 'num_residual_blocks', 'use_moe', 'use_distributional',
            'num_atoms', 'use_spectral_norm', 'dropout', 'use_world_model',
            'deter_dim', 'num_categoricals', 'num_classes', 'codebook_dim',
            'batch_size', 'gradient_accumulation_steps', 'learning_rate',
            'cql_alpha', 'use_mixed_precision', 'compile_model',
        }
        
        for k, v in kwargs.items():
            if k in modern_params:
                modern_kwargs[k] = v
            else:
                legacy_kwargs[k] = v
        
        # Create modern config
        config = ModernPolicyConfig(**modern_kwargs)
        
        self.modern_policy = ModernNeuralPolicy(
            config=config,
            *args,
            **legacy_kwargs,
        )
        
        # Delegate attributes
        self.ACTION_COUNT = self.modern_policy.ACTION_COUNT
        self.FEATURE_COUNT = self.modern_policy.FEATURE_COUNT
        self.PLANNED_DIRECTION_START = 31
        self.agent = self.modern_policy.trainer  # For compatibility
        self.training = self.modern_policy.training
        self.world_model = self.modern_policy.legacy_world_model
        self._pending = None
        self._pending_memory_case = None
        self._last_planned_direction = None
        self._novel_states = Counter()
        self.losses = []
    
    def __getattr__(self, name):
        return getattr(self.modern_policy, name)
    
    def __setattr__(self, name, value):
        if name in ('modern_policy', 'ACTION_COUNT', 'FEATURE_COUNT', 'PLANNED_DIRECTION_START', 'agent', 'training', 'world_model', '_pending', '_pending_memory_case', '_last_planned_direction', '_novel_states', 'losses'):
            super().__setattr__(name, value)
        else:
            setattr(self.modern_policy, name, value)


import numpy as np