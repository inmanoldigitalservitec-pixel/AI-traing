from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - handled by CLI at runtime.
    raise ImportError(
        "PyTorch no esta instalado. Instala con: python3 -m pip install torch"
    ) from exc

from .agent import TAP, WAIT
from .env import ColorJumpEnv, GameConfig, State


STATE_SIZE = 14
ACTION_SIZE = 2
MODEL_VERSION = 2


@dataclass
class DQNStats:
    episodes: int
    best_score: int
    average_score: float
    eval_average_score: float | None = None
    eval_best_score: int | None = None


@dataclass
class DQNConfig:
    learning_rate: float = 0.001
    gamma: float = 0.94
    epsilon_start: float = 0.35
    epsilon_end: float = 0.03
    batch_size: int = 64
    replay_size: int = 50_000
    warmup_steps: int = 500
    target_update_every: int = 250
    hidden_size: int = 96


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.items = deque(maxlen=capacity)

    def push(self, transition: tuple[list[float], int, float, list[float], bool]) -> None:
        self.items.append(transition)

    def sample(self, size: int):
        return random.sample(self.items, size)

    def __len__(self) -> int:
        return len(self.items)


class QNetwork(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(STATE_SIZE, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, ACTION_SIZE),
        )

    def forward(self, x):
        return self.layers(x)


class DQNAgent:
    def __init__(
        self,
        config: DQNConfig | None = None,
        seed: int | None = None,
        device: str | None = None,
    ):
        self.config = config or DQNConfig()
        self.random = random.Random(seed)
        torch.manual_seed(seed or 0)
        self.device = torch.device(device or ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.policy = QNetwork(self.config.hidden_size).to(self.device)
        self.target = QNetwork(self.config.hidden_size).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.config.learning_rate)
        self.replay = ReplayBuffer(self.config.replay_size)
        self.learn_steps = 0

    def choose_action(self, state: State, explore: bool = True, epsilon: float = 0.0) -> int:
        if explore and self.random.random() < epsilon:
            return self.random.choice([WAIT, TAP])

        with torch.no_grad():
            tensor = torch.tensor([state_to_vector(state)], dtype=torch.float32, device=self.device)
            q_values = self.policy(tensor)[0]
            wait_q, tap_q = float(q_values[WAIT].item()), float(q_values[TAP].item())

        if abs(tap_q - wait_q) < 0.75:
            return self._heuristic_action(state)

        return TAP if tap_q > wait_q else WAIT

    def train(
        self,
        episodes: int,
        seed: int | None = None,
        game_config: GameConfig | None = None,
        eval_episodes: int = 50,
    ) -> DQNStats:
        scores: list[int] = []
        best_score = 0

        for episode in range(episodes):
            progress = episode / max(1, episodes - 1)
            epsilon = self.config.epsilon_start + (
                self.config.epsilon_end - self.config.epsilon_start
            ) * progress

            env_seed = None if seed is None else seed + episode
            env = ColorJumpEnv(config=game_config, seed=env_seed)
            state = env.reset()
            done = False

            while not done:
                action = self.choose_action(state, explore=True, epsilon=epsilon)
                result = env.step(action)
                reward = self._shape_reward(state, action, result.reward, result.state)
                self.replay.push(
                    (
                        state_to_vector(state),
                        action,
                        reward,
                        state_to_vector(result.state),
                        result.done,
                    )
                )
                self._learn_step()
                state = result.state
                done = result.done

            scores.append(env.score)
            best_score = max(best_score, env.score)

        eval_stats = self.evaluate(
            eval_episodes,
            seed=None if seed is None else seed + episodes + 20_000,
            game_config=game_config,
        )
        return DQNStats(
            episodes=episodes,
            best_score=best_score,
            average_score=sum(scores) / len(scores) if scores else 0.0,
            eval_average_score=eval_stats.average_score,
            eval_best_score=eval_stats.best_score,
        )

    def evaluate(
        self,
        episodes: int,
        seed: int | None = None,
        game_config: GameConfig | None = None,
    ) -> DQNStats:
        scores: list[int] = []
        best_score = 0

        for episode in range(episodes):
            env_seed = None if seed is None else seed + episode
            env = ColorJumpEnv(config=game_config, seed=env_seed)
            state = env.reset()
            done = False

            while not done:
                action = self.choose_action(state, explore=False)
                result = env.step(action)
                state = result.state
                done = result.done

            scores.append(env.score)
            best_score = max(best_score, env.score)

        return DQNStats(
            episodes=episodes,
            best_score=best_score,
            average_score=sum(scores) / len(scores) if scores else 0.0,
        )

    def save(self, path: str) -> None:
        torch.save(
            {
                "model": self.policy.state_dict(),
                "config": self.config.__dict__,
                "model_version": MODEL_VERSION,
                "state_size": STATE_SIZE,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str | None = None) -> DQNAgent:
        checkpoint = torch.load(path, map_location=device or "cpu")
        state_size = int(checkpoint.get("state_size", 0))
        model_version = int(checkpoint.get("model_version", 1))
        if state_size != STATE_SIZE or model_version != MODEL_VERSION:
            raise ValueError(
                "Modelo DQN incompatible con esta version. "
                "Borra el modelo viejo con: python3 -m color_jump_ai dqn-reset"
            )
        agent = cls(config=DQNConfig(**checkpoint["config"]), device=device)
        agent.policy.load_state_dict(checkpoint["model"])
        agent.target.load_state_dict(agent.policy.state_dict())
        return agent

    def _heuristic_action(self, state: State) -> int:
        distance = state.distance_bucket
        velocity = state.velocity_bucket
        color_matches = state.ball_color == state.target_color

        if state.tap_ready == 0:
            return WAIT

        if not color_matches and distance <= 260:
            return WAIT

        if color_matches and 0 <= distance <= 240:
            return TAP

        if velocity <= -3 and distance > 100:
            return TAP

        if 0 <= distance <= 220:
            if not color_matches:
                return WAIT
            if velocity <= 5:
                return TAP
            return WAIT

        if distance <= 0 and color_matches and velocity < 6:
            return TAP

        return WAIT

    def _shape_reward(self, state: State, action: int, reward: float, next_state: State) -> float:
        shaped = reward

        if next_state.falling and next_state.velocity_bucket <= -6:
            shaped -= 0.08

        if state.tap_ready and action == WAIT and state.velocity_bucket <= -4:
            shaped -= 0.05

        if state.tap_ready and action == TAP and state.ball_color == state.target_color:
            shaped += 0.03

        if state.tap_ready and action == TAP and state.ball_color != state.target_color:
            shaped -= 0.03

        return shaped

    def _learn_step(self) -> None:
        if len(self.replay) < max(self.config.batch_size, self.config.warmup_steps):
            return

        batch = self.replay.sample(self.config.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        current_q = self.policy(states_t).gather(1, actions_t)
        with torch.no_grad():
            next_q = self.target(next_states_t).max(dim=1, keepdim=True).values
            target_q = rewards_t + self.config.gamma * next_q * (1.0 - dones_t)

        loss = nn.functional.smooth_l1_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()

        self.learn_steps += 1
        if self.learn_steps % self.config.target_update_every == 0:
            self.target.load_state_dict(self.policy.state_dict())


def state_to_vector(state: State) -> list[float]:
    return [
        max(-1.0, min(1.0, state.distance_bucket / 500.0)),
        max(-1.0, min(1.0, state.velocity_bucket / 20.0)),
        1.0 if state.ball_color == 0 else 0.0,
        1.0 if state.ball_color == 1 else 0.0,
        1.0 if state.ball_color == 2 else 0.0,
        1.0 if state.ball_color == 3 else 0.0,
        1.0 if state.target_color == 0 else 0.0,
        1.0 if state.target_color == 1 else 0.0,
        1.0 if state.target_color == 2 else 0.0,
        1.0 if state.target_color == 3 else 0.0,
        1.0 if state.target_color == state.ball_color else 0.0,
        1.0 if state.falling else 0.0,
        1.0 if state.tap_ready else 0.0,
        max(0.0, min(1.0, abs(state.distance_bucket) / 500.0)),
    ]
