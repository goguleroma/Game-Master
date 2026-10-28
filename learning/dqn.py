from collections import deque
from dataclasses import dataclass
import math
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class _Transition:
    state: tuple[float, ...]
    action: int
    reward: float
    next_state: tuple[float, ...]
    next_action_mask: tuple[bool, ...]
    done: bool


class _QNetwork(nn.Module):
    def __init__(self, input_size: int, action_count: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_count),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.layers(state)


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
        self.input_size = input_size
        self.action_count = action_count
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.warmup_steps = warmup_steps
        self.target_update_steps = target_update_steps
        self._rng = random.Random(seed)
        torch.manual_seed(seed)
        self.online = _QNetwork(input_size, action_count)
        self.target = _QNetwork(input_size, action_count)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=learning_rate)
        self.replay: deque[_Transition] = deque(maxlen=replay_capacity)
        self.environment_steps = 0
        self.training_steps = 0

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
                values = self.online(torch.tensor(state, dtype=torch.float32))
                values = values.masked_fill(
                    ~torch.tensor(action_mask, dtype=torch.bool),
                    -torch.inf,
                )
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
        batch = self._rng.sample(self.replay, self.batch_size)
        states = torch.tensor(
            np.asarray([item.state for item in batch]),
            dtype=torch.float32,
        )
        actions = torch.tensor(
            [item.action for item in batch],
            dtype=torch.int64,
        )
        rewards = torch.tensor(
            [item.reward for item in batch],
            dtype=torch.float32,
        )
        next_states = torch.tensor(
            np.asarray([item.next_state for item in batch]),
            dtype=torch.float32,
        )
        next_masks = torch.tensor(
            np.asarray([item.next_action_mask for item in batch]),
            dtype=torch.bool,
        )
        dones = torch.tensor([item.done for item in batch], dtype=torch.float32)

        current_values = self.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            online_next = self.online(next_states).masked_fill(
                ~next_masks,
                -torch.inf,
            )
            next_actions = online_next.argmax(dim=1)
            target_next = self.target(next_states).gather(
                1,
                next_actions.unsqueeze(1),
            ).squeeze(1)
            target_next = torch.where(dones.bool(), 0.0, target_next)
            expected_values = rewards + self.gamma * target_next

        loss = nn.functional.smooth_l1_loss(current_values, expected_values)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.training_steps += 1
        if self.training_steps % self.target_update_steps == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())

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
                )
                actions = torch.tensor(
                    [sample[1] for sample in batch],
                    dtype=torch.int64,
                )
                action_masks = torch.tensor(
                    np.asarray([sample[2] for sample in batch]),
                    dtype=torch.bool,
                )
                logits = self.online(states).masked_fill(
                    ~action_masks,
                    -torch.inf,
                )
                loss = nn.functional.cross_entropy(logits, actions)
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
        payload: dict[str, Any] = {
            "format_version": self.FORMAT_VERSION,
            "input_size": self.input_size,
            "action_count": self.action_count,
            "online_state_dict": self.online.state_dict(),
            "target_state_dict": self.target.state_dict(),
            "environment_steps": self.environment_steps,
            "training_steps": self.training_steps,
            "metadata": metadata or {},
            "has_training_state": include_training_state,
        }
        if include_training_state:
            payload.update(
                {
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "replay_capacity": self.replay.maxlen,
                    "replay": [
                        (
                            item.state,
                            item.action,
                            item.reward,
                            item.next_state,
                            item.next_action_mask,
                            item.done,
                        )
                        for item in self.replay
                    ],
                    "python_rng_state": self._rng.getstate(),
                    "training_config": {
                        "batch_size": self.batch_size,
                        "gamma": self.gamma,
                        "epsilon_decay_steps": self.epsilon_decay_steps,
                        "epsilon_start": self.epsilon_start,
                        "epsilon_end": self.epsilon_end,
                        "warmup_steps": self.warmup_steps,
                        "target_update_steps": self.target_update_steps,
                    },
                }
            )
        temporary_path = path.with_name(f"{path.name}.tmp")
        torch.save(payload, temporary_path)
        temporary_path.replace(path)

    def load(self, path: Path, resume_training: bool = True) -> dict[str, Any]:
        payload = torch.load(path, map_location="cpu", weights_only=True)
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
            }
            expected_config = {
                "batch_size": self.batch_size,
                "gamma": self.gamma,
                "epsilon_decay_steps": self.epsilon_decay_steps,
                "epsilon_start": self.epsilon_start,
                "epsilon_end": self.epsilon_end,
                "warmup_steps": self.warmup_steps,
                "target_update_steps": self.target_update_steps,
            }
            if normalized_config != expected_config:
                raise ValueError(
                    "Checkpoint training configuration does not match this DQN agent."
                )
            replay_capacity = payload.get("replay_capacity")
            replay = payload.get("replay")
            rng_state = payload.get("python_rng_state")
            optimizer_state = payload.get("optimizer_state_dict")
            if (
                not isinstance(replay_capacity, int)
                or replay_capacity <= 0
                or replay_capacity != self.replay.maxlen
                or not isinstance(replay, list)
                or not isinstance(rng_state, tuple)
                or not isinstance(optimizer_state, dict)
            ):
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
