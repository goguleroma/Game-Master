"""
World Model (RSSM - Recurrent State Space Model) for Model-Based RL.

Based on DreamerV3 architecture:
- Discrete Autoencoder for observations (VQ-VAE style)
- RSSM: Recurrent State Space Model with discrete latent states
- Imagination rollouts for policy learning
- Uncertainty estimation (ensemble disagreement + aleatoric)
"""
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
from torch import nn
from torch.nn import functional as F

from learning.networks import (
    ResNetBackbone,
    spectral_norm,
    apply_spectral_norm,
    SwiGLUBlock,
)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class WorldModelConfig:
    # Observation
    obs_dim: int = 40
    obs_embed_dim: int = 256
    
    # Discrete Autoencoder (VQ-VAE)
    num_categoricals: int = 32
    num_classes: int = 32
    codebook_dim: int = 256
    
    # RSSM
    deter_dim: int = 512  # Deterministic recurrent state
    stoch_dim: int = 1024  # Stochastic state (num_categoricals * num_classes)
    hidden_dim: int = 512
    num_layers: int = 2
    dropout: float = 0.1
    
    # Dynamics
    use_spectral_norm: bool = True
    
    # Uncertainty
    ensemble_size: int = 5
    
    # Reconstruction
    recon_loss_weight: float = 1.0
    kl_loss_weight: float = 0.1
    kl_free_nats: float = 1.0
    
    # Imagination
    imagination_horizon: int = 15
    imagination_batch: int = 64


# ============================================================================
# Discrete Autoencoder (VQ-VAE style)
# ============================================================================

class DiscreteEncoder(nn.Module):
    """
    Encodes observations to discrete latent codes.
    Uses straight-through Gumbel-Softmax for differentiable sampling.
    """
    def __init__(
        self,
        obs_dim: int,
        embed_dim: int,
        num_categoricals: int,
        num_classes: int,
        hidden_dim: int = 512,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_spectral_norm: bool = True,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.num_categoricals = num_categoricals
        self.num_classes = num_classes
        self.temperature = temperature
        
        # Encoder network
        layers = [
            spectral_norm(nn.Linear(obs_dim, hidden_dim, bias=False)) if use_spectral_norm else nn.Linear(obs_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        ]
        for _ in range(num_layers - 1):
            layers.extend([
                spectral_norm(nn.Linear(hidden_dim, hidden_dim, bias=False)) if use_spectral_norm else nn.Linear(hidden_dim, hidden_dim, bias=False),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
            ])
        layers.append(
            spectral_norm(nn.Linear(hidden_dim, num_categoricals * num_classes, bias=False)) if use_spectral_norm else nn.Linear(hidden_dim, num_categoricals * num_classes, bias=False)
        )
        self.encoder = nn.Sequential(*layers)
        
        # Codebook (embedding table)
        self.codebook = nn.Embedding(num_categoricals * num_classes, embed_dim)
        
    def forward(self, obs: torch.Tensor, hard: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            obs: (batch, obs_dim)
        Returns:
            z: (batch, num_categoricals, embed_dim) - discrete latent embeddings
            logits: (batch, num_categoricals, num_classes) - pre-softmax logits
            probs: (batch, num_categoricals, num_classes) - softmax probabilities
        """
        batch_size = obs.shape[0]
        logits = self.encoder(obs).view(batch_size, self.num_categoricals, self.num_classes)
        probs = F.softmax(logits / self.temperature, dim=-1)
        
        if self.training or not hard:
            # Gumbel-Softmax (differentiable)
            gumbel = -torch.log(-torch.log(torch.rand_like(probs) + 1e-8) + 1e-8)
            sample = F.softmax((logits + gumbel) / self.temperature, dim=-1)
        else:
            # Hard one-hot (for inference)
            indices = probs.argmax(dim=-1)
            sample = F.one_hot(indices, self.num_classes).float()
            # Straight-through estimator
            sample = sample - probs.detach() + probs
        
        # Embed to continuous space
        # sample: (batch, num_categoricals, num_classes)
        # codebook: (num_categoricals * num_classes, embed_dim)
        flat_indices = sample.argmax(dim=-1)  # (batch, num_categoricals)
        z = self.codebook(flat_indices)  # (batch, num_categoricals, embed_dim)
        
        return z, logits, probs
    
    def get_codebook_loss(self) -> torch.Tensor:
        """Codebook commitment loss (not needed for straight-through)."""
        return torch.tensor(0.0, device=self.codebook.weight.device)


class DiscreteDecoder(nn.Module):
    """Decodes discrete latent codes back to observations."""
    def __init__(
        self,
        embed_dim: int,
        obs_dim: int,
        num_categoricals: int,
        hidden_dim: int = 512,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_spectral_norm: bool = True,
    ):
        super().__init__()
        self.num_categoricals = num_categoricals
        input_dim = num_categoricals * embed_dim
        
        layers = [
            spectral_norm(nn.Linear(input_dim, hidden_dim, bias=False)) if use_spectral_norm else nn.Linear(input_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        ]
        for _ in range(num_layers - 1):
            layers.extend([
                spectral_norm(nn.Linear(hidden_dim, hidden_dim, bias=False)) if use_spectral_norm else nn.Linear(hidden_dim, hidden_dim, bias=False),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
            ])
        layers.append(
            spectral_norm(nn.Linear(hidden_dim, obs_dim, bias=False)) if use_spectral_norm else nn.Linear(hidden_dim, obs_dim, bias=False)
        )
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (batch, num_categoricals, embed_dim)
        Returns:
            obs_recon: (batch, obs_dim)
        """
        batch_size = z.shape[0]
        z_flat = z.view(batch_size, -1)
        return self.decoder(z_flat)


# ============================================================================
# RSSM (Recurrent State Space Model)
# ============================================================================

class RSSMPrior(nn.Module):
    """
    Prior network: p(z_t | h_t)
    Predicts stochastic state from deterministic recurrent state.
    """
    def __init__(
        self,
        deter_dim: int,
        stoch_dim: int,
        hidden_dim: int,
        num_categoricals: int,
        num_classes: int,
        use_spectral_norm: bool = True,
    ):
        super().__init__()
        self.num_categoricals = num_categoricals
        self.num_classes = num_classes
        
        self.net = nn.Sequential(
            spectral_norm(nn.Linear(deter_dim, hidden_dim, bias=False)) if use_spectral_norm else nn.Linear(deter_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            spectral_norm(nn.Linear(hidden_dim, num_categoricals * num_classes, bias=False)) if use_spectral_norm else nn.Linear(hidden_dim, num_categoricals * num_classes, bias=False),
        )
    
    def forward(self, deter: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            deter: (batch, deter_dim)
        Returns:
            logits: (batch, num_categoricals, num_classes)
            probs: (batch, num_categoricals, num_classes)
        """
        logits = self.net(deter).view(-1, self.num_categoricals, self.num_classes)
        probs = F.softmax(logits, dim=-1)
        return logits, probs


class RSSMPosterior(nn.Module):
    """
    Posterior network: q(z_t | h_t, x_t)
    Encodes observation into stochastic state given deterministic state.
    """
    def __init__(
        self,
        deter_dim: int,
        embed_dim: int,  # This is now num_categoricals * codebook_dim (flattened z_embed)
        stoch_dim: int,
        hidden_dim: int,
        num_categoricals: int,
        num_classes: int,
        use_spectral_norm: bool = True,
    ):
        super().__init__()
        self.num_categoricals = num_categoricals
        self.num_classes = num_classes
        
        self.net = nn.Sequential(
            spectral_norm(nn.Linear(deter_dim + embed_dim, hidden_dim, bias=False)) if use_spectral_norm else nn.Linear(deter_dim + embed_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            spectral_norm(nn.Linear(hidden_dim, num_categoricals * num_classes, bias=False)) if use_spectral_norm else nn.Linear(hidden_dim, num_categoricals * num_classes, bias=False),
        )
    
    def forward(self, deter: torch.Tensor, embed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            deter: (batch, deter_dim)
            embed: (batch, embed_dim) - flattened encoded observation (num_categoricals * codebook_dim)
        Returns:
            logits: (batch, num_categoricals, num_classes)
            probs: (batch, num_categoricals, num_classes)
        """
        x = torch.cat([deter, embed], dim=-1)
        logits = self.net(x).view(-1, self.num_categoricals, self.num_classes)
        probs = F.softmax(logits, dim=-1)
        return logits, probs


class RSSMRecurrent(nn.Module):
    """
    Recurrent deterministic state update: h_t = f(h_{t-1}, z_{t-1}, a_{t-1})
    Uses GRU for recurrent dynamics.
    """
    def __init__(
        self,
        deter_dim: int,
        stoch_dim: int,
        action_dim: int,
        hidden_dim: int,
        use_spectral_norm: bool = True,
    ):
        super().__init__()
        self.deter_dim = deter_dim
        input_dim = stoch_dim + action_dim
        
        # GRU cell - input is the concatenated (stoch + action), hidden is deter_dim
        self.gru = nn.GRUCell(input_dim, deter_dim)
        
        # Optional input projection for deeper processing
        self.input_proj = nn.Sequential(
            spectral_norm(nn.Linear(input_dim, hidden_dim, bias=False)) if use_spectral_norm else nn.Linear(input_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            spectral_norm(nn.Linear(hidden_dim, input_dim, bias=False)) if use_spectral_norm else nn.Linear(hidden_dim, input_dim, bias=False),
        )
    
    def forward(
        self,
        deter_prev: torch.Tensor,
        stoch_prev: torch.Tensor,
        action_prev: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            deter_prev: (batch, deter_dim)
            stoch_prev: (batch, stoch_dim) - flattened discrete latent
            action_prev: (batch, action_dim) - one-hot action
        Returns:
            deter: (batch, deter_dim)
        """
        # Concatenate stochastic state and action
        x = torch.cat([stoch_prev, action_prev], dim=-1)
        # Process through projection (residual)
        x = x + self.input_proj(x)
        # GRU update
        deter = self.gru(x, deter_prev)
        return deter


class RSSM(nn.Module):
    """
    Full Recurrent State Space Model.
    Combines prior, posterior, and recurrent dynamics.
    """
    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.config = config
        stoch_dim = config.num_categoricals * config.num_classes
        
        self.recurrent = RSSMRecurrent(
            deter_dim=config.deter_dim,
            stoch_dim=stoch_dim,
            action_dim=config.num_categoricals,  # Will be overridden by actual action_dim
            hidden_dim=config.hidden_dim,
            use_spectral_norm=config.use_spectral_norm,
        )
        
        self.prior = RSSMPrior(
            deter_dim=config.deter_dim,
            stoch_dim=stoch_dim,
            hidden_dim=config.hidden_dim,
            num_categoricals=config.num_categoricals,
            num_classes=config.num_classes,
            use_spectral_norm=config.use_spectral_norm,
        )
        
        # embed_dim for posterior is flattened z_embed: num_categoricals * codebook_dim
        posterior_embed_dim = config.num_categoricals * config.codebook_dim
        
        self.posterior = RSSMPosterior(
            deter_dim=config.deter_dim,
            embed_dim=posterior_embed_dim,
            stoch_dim=stoch_dim,
            hidden_dim=config.hidden_dim,
            num_categoricals=config.num_categoricals,
            num_classes=config.num_classes,
            use_spectral_norm=config.use_spectral_norm,
        )
        
        # Discrete autoencoder
        self.encoder = DiscreteEncoder(
            obs_dim=config.obs_dim,
            embed_dim=config.codebook_dim,
            num_categoricals=config.num_categoricals,
            num_classes=config.num_classes,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            use_spectral_norm=config.use_spectral_norm,
        )
        
        self.decoder = DiscreteDecoder(
            embed_dim=config.codebook_dim,
            obs_dim=config.obs_dim,
            num_categoricals=config.num_categoricals,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            use_spectral_norm=config.use_spectral_norm,
        )
    
    def initial_state(self, batch_size: int, device: torch.device) -> dict:
        """Initialize recurrent state."""
        return {
            'deter': torch.zeros(batch_size, self.config.deter_dim, device=device),
            'stoch': torch.zeros(batch_size, self.config.num_categoricals * self.config.num_classes, device=device),
            'logits': torch.zeros(batch_size, self.config.num_categoricals, self.config.num_classes, device=device),
        }
    
    def observe(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        state: Optional[dict] = None,
    ) -> Tuple[dict, dict]:
        """
        Encode sequence of observations (posterior inference).
        
        Args:
            obs: (seq_len, batch, obs_dim)
            actions: (seq_len, batch, action_dim) - one-hot
            state: initial state dict
        Returns:
            posterior_states: dict with deter, stoch, logits, probs, embed
            prior_states: dict with deter, stoch, logits, probs
        """
        seq_len, batch_size = obs.shape[:2]
        device = obs.device
        
        if state is None:
            state = self.initial_state(batch_size, device)
        
        deter = state['deter']
        stoch = state['stoch']
        
        posteriors = {'deter': [], 'stoch': [], 'logits': [], 'probs': [], 'embed': []}
        priors = {'deter': [], 'stoch': [], 'logits': [], 'probs': []}
        
        for t in range(seq_len):
            # Recurrent update
            deter = self.recurrent(deter, stoch, actions[t])
            
            # Prior
            prior_logits, prior_probs = self.prior(deter)
            prior_stoch = self._sample_stochastic(prior_probs)
            
            # Posterior
            z_embed, _, _ = self.encoder(obs[t])
            embed_flat = z_embed.view(batch_size, -1)
            post_logits, post_probs = self.posterior(deter, embed_flat)
            post_stoch = self._sample_stochastic(post_probs)
            
            # Store
            posteriors['deter'].append(deter)
            posteriors['stoch'].append(post_stoch)
            posteriors['logits'].append(post_logits)
            posteriors['probs'].append(post_probs)
            posteriors['embed'].append(z_embed)
            
            priors['deter'].append(deter)
            priors['stoch'].append(prior_stoch)
            priors['logits'].append(prior_logits)
            priors['probs'].append(prior_probs)
            
            # Update for next step
            stoch = post_stoch
        
        # Stack sequences
        for k in posteriors:
            posteriors[k] = torch.stack(posteriors[k], dim=0)
        for k in priors:
            priors[k] = torch.stack(priors[k], dim=0)
        
        return posteriors, priors
    
    def imagine(
        self,
        policy: nn.Module,
        state: dict,
        horizon: int,
        action_dim: int,
    ) -> dict:
        """
        Imagination rollout: generate trajectory using policy.
        
        Args:
            policy: policy network (takes deter+stoch, outputs action logits)
            state: initial state dict
            horizon: rollout length
            action_dim: number of actions
        Returns:
            imagined trajectory dict
        """
        batch_size = state['deter'].shape[0]
        device = state['deter'].device
        
        deter = state['deter']
        stoch = state['stoch']
        
        traj = {
            'deter': [],
            'stoch': [],
            'logits': [],
            'probs': [],
            'actions': [],
            'rewards': [],
        }
        
        # Ensure horizon is at least 1
        horizon = max(1, horizon)
        
        for t in range(horizon):
            # Policy action
            policy_input = torch.cat([deter, stoch], dim=-1)
            action_logits = policy(policy_input)
            action_dist = torch.distributions.Categorical(logits=action_logits)
            action = action_dist.sample()
            action_onehot = F.one_hot(action, action_dim).float()
            
            # Recurrent update
            deter = self.recurrent(deter, stoch, action_onehot)
            
            # Prior
            prior_logits, prior_probs = self.prior(deter)
            stoch = self._sample_stochastic(prior_probs)
            
            # Store
            traj['deter'].append(deter)
            traj['stoch'].append(stoch)
            traj['logits'].append(prior_logits)
            traj['probs'].append(prior_probs)
            traj['actions'].append(action_onehot)
        
        # Stack trajectory
        for k in traj:
            if len(traj[k]) > 0:
                traj[k] = torch.stack(traj[k], dim=0)
            else:
                # Create empty tensor with correct shape
                if k == 'deter':
                    traj[k] = torch.zeros(0, batch_size, self.config.deter_dim, device=device)
                elif k == 'stoch':
                    traj[k] = torch.zeros(0, batch_size, self.config.num_categoricals * self.config.num_classes, device=device)
                elif k == 'logits':
                    traj[k] = torch.zeros(0, batch_size, self.config.num_categoricals, self.config.num_classes, device=device)
                elif k == 'probs':
                    traj[k] = torch.zeros(0, batch_size, self.config.num_categoricals, self.config.num_classes, device=device)
                elif k == 'actions':
                    traj[k] = torch.zeros(0, batch_size, action_dim, device=device)
                elif k == 'rewards':
                    traj[k] = torch.zeros(0, batch_size, device=device)
        
        return traj
    
    def _sample_stochastic(self, probs: torch.Tensor) -> torch.Tensor:
        """Sample stochastic state from categorical distribution."""
        if self.training:
            # Gumbel-Softmax
            gumbel = -torch.log(-torch.log(torch.rand_like(probs) + 1e-8) + 1e-8)
            sample = F.softmax((probs.log() + gumbel), dim=-1)
        else:
            indices = probs.argmax(dim=-1)
            sample = F.one_hot(indices, self.config.num_classes).float()
        return sample.view(probs.shape[0], -1)
    
    def decode(self, stoch: torch.Tensor) -> torch.Tensor:
        """Decode stochastic state to observation."""
        batch_size = stoch.shape[0]
        z = stoch.view(batch_size, self.config.num_categoricals, -1)
        # Find indices from one-hot
        indices = z.argmax(dim=-1)
        z_embed = self.encoder.codebook(indices)
        return self.decoder(z_embed)
    
    def compute_losses(
        self,
        posteriors: dict,
        priors: dict,
        obs: torch.Tensor,
        free_nats: float = 1.0,
    ) -> dict:
        """
        Compute world model losses.
        
        Args:
            posteriors: from observe()
            priors: from observe()
            obs: (seq_len, batch, obs_dim) - ground truth observations
        Returns:
            loss dict
        """
        seq_len, batch_size = obs.shape[:2]
        
        # Reconstruction loss
        recon_loss = 0.0
        for t in range(seq_len):
            z_embed = posteriors['embed'][t]
            obs_recon = self.decoder(z_embed)
            recon_loss += F.mse_loss(obs_recon, obs[t])
        recon_loss = recon_loss / seq_len
        
        # KL loss (posterior || prior)
        kl_loss = 0.0
        for t in range(seq_len):
            post_probs = posteriors['probs'][t]  # (batch, cat, classes)
            prior_probs = priors['probs'][t]
            
            # KL per categorical
            kl = post_probs * (post_probs.log() - prior_probs.log() + 1e-8)
            kl = kl.sum(dim=-1).mean()  # sum over classes, mean over categoricals and batch
            kl_loss += kl
        kl_loss = kl_loss / seq_len
        
        # Free nats constraint
        kl_loss = torch.clamp(kl_loss, min=free_nats)
        
        return {
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'total_loss': recon_loss + kl_loss,
        }


# ============================================================================
# Reward & Continue Predictors
# ============================================================================

class RewardPredictor(nn.Module):
    """Predicts reward from deterministic + stochastic state."""
    def __init__(
        self,
        deter_dim: int,
        stoch_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_spectral_norm: bool = True,
        ensemble_size: int = 1,
    ):
        super().__init__()
        self.ensemble_size = ensemble_size
        
        if ensemble_size > 1:
            self.nets = nn.ModuleList([
                self._make_net(deter_dim + stoch_dim, hidden_dim, num_layers, dropout, use_spectral_norm)
                for _ in range(ensemble_size)
            ])
        else:
            self.net = self._make_net(deter_dim + stoch_dim, hidden_dim, num_layers, dropout, use_spectral_norm)
    
    def _make_net(self, input_dim, hidden_dim, num_layers, dropout, use_spectral_norm):
        layers = [
            spectral_norm(nn.Linear(input_dim, hidden_dim, bias=False)) if use_spectral_norm else nn.Linear(input_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        ]
        for _ in range(num_layers - 2):
            layers.extend([
                spectral_norm(nn.Linear(hidden_dim, hidden_dim, bias=False)) if use_spectral_norm else nn.Linear(hidden_dim, hidden_dim, bias=False),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
            ])
        layers.append(
            spectral_norm(nn.Linear(hidden_dim, 1, bias=False)) if use_spectral_norm else nn.Linear(hidden_dim, 1, bias=False)
        )
        return nn.Sequential(*layers)
    
    def forward(self, deter: torch.Tensor, stoch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            deter: (batch, deter_dim)
            stoch: (batch, stoch_dim)
        Returns:
            reward: (batch,) or (ensemble, batch) if ensemble
        """
        x = torch.cat([deter, stoch], dim=-1)
        if self.ensemble_size > 1:
            return torch.stack([net(x).squeeze(-1) for net in self.nets], dim=0)
        else:
            return self.net(x).squeeze(-1)
    
    def get_uncertainty(self, deter: torch.Tensor, stoch: torch.Tensor) -> torch.Tensor:
        """Get ensemble disagreement (epistemic uncertainty)."""
        if self.ensemble_size <= 1:
            return torch.zeros(deter.shape[0], device=deter.device)
        preds = self.forward(deter, stoch)  # (ensemble, batch)
        return preds.std(dim=0)


class ContinuePredictor(nn.Module):
    """Predicts episode continuation probability."""
    def __init__(
        self,
        deter_dim: int,
        stoch_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_spectral_norm: bool = True,
    ):
        super().__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Linear(deter_dim + stoch_dim, hidden_dim, bias=False)) if use_spectral_norm else nn.Linear(deter_dim + stoch_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            spectral_norm(nn.Linear(hidden_dim, 1, bias=False)) if use_spectral_norm else nn.Linear(hidden_dim, 1, bias=False),
        )
    
    def forward(self, deter: torch.Tensor, stoch: torch.Tensor) -> torch.Tensor:
        x = torch.cat([deter, stoch], dim=-1)
        return torch.sigmoid(self.net(x).squeeze(-1))


# ============================================================================
# Uncertainty Estimation
# ============================================================================

class UncertaintyHead(nn.Module):
    """
    Uncertainty estimation head.
    Combines ensemble disagreement (epistemic) + aleatoric uncertainty.
    """
    def __init__(
        self,
        deter_dim: int,
        stoch_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_spectral_norm: bool = True,
    ):
        super().__init__()
        # Aleatoric uncertainty (learned variance)
        self.aleatoric_net = nn.Sequential(
            spectral_norm(nn.Linear(deter_dim + stoch_dim, hidden_dim, bias=False)) if use_spectral_norm else nn.Linear(deter_dim + stoch_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            spectral_norm(nn.Linear(hidden_dim, 1, bias=False)) if use_spectral_norm else nn.Linear(hidden_dim, 1, bias=False),
        )
        # Softplus for positive variance
        self.softplus = nn.Softplus()
    
    def forward(self, deter: torch.Tensor, stoch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            aleatoric_var: (batch,) - learned variance
            total_uncertainty: (batch,) - combined (needs epistemic from ensemble)
        """
        x = torch.cat([deter, stoch], dim=-1)
        aleatoric_var = self.softplus(self.aleatoric_net(x).squeeze(-1)) + 1e-4
        return aleatoric_var, aleatoric_var  # total = aleatoric if no ensemble


# ============================================================================
# World Model Wrapper (for policy learning)
# ============================================================================

class WorldModel(nn.Module):
    """
    Complete World Model for model-based RL.
    Integrates RSSM, reward/continue predictors, and uncertainty.
    """
    def __init__(self, config: WorldModelConfig, action_dim: int):
        super().__init__()
        self.config = config
        self.action_dim = action_dim
        
        # Update RSSM recurrent with correct action_dim
        stoch_dim = config.num_categoricals * config.num_classes
        self.rssm = RSSM(config)
        self.rssm.recurrent = RSSMRecurrent(
            deter_dim=config.deter_dim,
            stoch_dim=stoch_dim,
            action_dim=action_dim,
            hidden_dim=config.hidden_dim,
            use_spectral_norm=config.use_spectral_norm,
        )
        
        # Reward predictor (ensemble for uncertainty)
        self.reward_predictor = RewardPredictor(
            deter_dim=config.deter_dim,
            stoch_dim=stoch_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            use_spectral_norm=config.use_spectral_norm,
            ensemble_size=config.ensemble_size,
        )
        
        # Continue predictor
        self.continue_predictor = ContinuePredictor(
            deter_dim=config.deter_dim,
            stoch_dim=stoch_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            use_spectral_norm=config.use_spectral_norm,
        )
        
        # Uncertainty head (aleatoric)
        self.uncertainty_head = UncertaintyHead(
            deter_dim=config.deter_dim,
            stoch_dim=stoch_dim,
            hidden_dim=config.hidden_dim // 2,
            num_layers=2,
            dropout=config.dropout,
            use_spectral_norm=config.use_spectral_norm,
        )
    
    def observe(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        continues: torch.Tensor,
        state: Optional[dict] = None,
    ) -> dict:
        """
        Process observation sequence (training).
        
        Returns dict with losses and states.
        """
        posteriors, priors = self.rssm.observe(obs, actions, state)
        
        # World model losses
        wm_losses = self.rssm.compute_losses(
            posteriors, priors, obs,
            free_nats=self.config.kl_free_nats
        )
        
        # Reward prediction loss
        seq_len, batch_size = obs.shape[:2]
        reward_loss = 0.0
        for t in range(seq_len):
            deter = posteriors['deter'][t]
            stoch = posteriors['stoch'][t]
            pred_rewards = self.reward_predictor(deter, stoch)
            if self.config.ensemble_size > 1:
                # Average over ensemble
                reward_loss += F.mse_loss(pred_rewards.mean(0), rewards[t])
            else:
                reward_loss += F.mse_loss(pred_rewards, rewards[t])
        reward_loss = reward_loss / seq_len
        
        # Continue prediction loss
        continue_loss = 0.0
        for t in range(seq_len):
            deter = posteriors['deter'][t]
            stoch = posteriors['stoch'][t]
            pred_continue = self.continue_predictor(deter, stoch)
            continue_loss += F.binary_cross_entropy(pred_continue, continues[t])
        continue_loss = continue_loss / seq_len
        
        # Total loss
        total_loss = (
            wm_losses['total_loss'] * self.config.recon_loss_weight +
            reward_loss +
            continue_loss
        )
        
        return {
            'posteriors': posteriors,
            'priors': priors,
            'wm_losses': wm_losses,
            'reward_loss': reward_loss,
            'continue_loss': continue_loss,
            'total_loss': total_loss,
        }
    
    def imagine(
        self,
        policy: nn.Module,
        state: dict,
        horizon: Optional[int] = None,
    ) -> dict:
        """
        Imagination rollout for policy learning.
        
        Returns trajectory with predicted rewards, continues, uncertainties.
        """
        horizon = horizon or self.config.imagination_horizon
        traj = self.rssm.imagine(policy, state, horizon, self.action_dim)
        
        # Predict rewards, continues, uncertainties along trajectory
        seq_len = traj['deter'].shape[0]
        batch_size = traj['deter'].shape[1]
        
        rewards = []
        continues = []
        aleatoric_vars = []
        epistemic_vars = []
        
        for t in range(seq_len):
            deter = traj['deter'][t]
            stoch = traj['stoch'][t]
            
            # Reward (ensemble)
            pred_rewards = self.reward_predictor(deter, stoch)
            if self.config.ensemble_size > 1:
                rewards.append(pred_rewards.mean(0))
                epistemic_vars.append(pred_rewards.std(0))
            else:
                rewards.append(pred_rewards)
                epistemic_vars.append(torch.zeros_like(pred_rewards))
            
            # Continue
            pred_continue = self.continue_predictor(deter, stoch)
            continues.append(pred_continue)
            
            # Aleatoric uncertainty
            aleatoric_var, _ = self.uncertainty_head(deter, stoch)
            aleatoric_vars.append(aleatoric_var)
        
        traj['rewards'] = torch.stack(rewards, dim=0)
        traj['continues'] = torch.stack(continues, dim=0)
        traj['aleatoric_var'] = torch.stack(aleatoric_vars, dim=0)
        traj['epistemic_var'] = torch.stack(epistemic_vars, dim=0)
        traj['total_uncertainty'] = traj['aleatoric_var'] + traj['epistemic_var']
        
        return traj
    
    def initial_state(self, batch_size: int, device: torch.device) -> dict:
        return self.rssm.initial_state(batch_size, device)


# ============================================================================
# Factory
# ============================================================================

def create_world_model(
    obs_dim: int = 40,
    action_dim: int = 6,
    deter_dim: int = 512,
    num_categoricals: int = 32,
    num_classes: int = 32,
    codebook_dim: int = 256,
    hidden_dim: int = 512,
    num_layers: int = 3,
    dropout: float = 0.1,
    use_spectral_norm: bool = True,
    ensemble_size: int = 5,
    imagination_horizon: int = 15,
) -> WorldModel:
    """Factory for creating world model with sensible defaults."""
    config = WorldModelConfig(
        obs_dim=obs_dim,
        deter_dim=deter_dim,
        num_categoricals=num_categoricals,
        num_classes=num_classes,
        codebook_dim=codebook_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        use_spectral_norm=use_spectral_norm,
        ensemble_size=ensemble_size,
        imagination_horizon=imagination_horizon,
    )
    return WorldModel(config, action_dim)