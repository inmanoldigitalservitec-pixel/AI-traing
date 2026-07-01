from __future__ import annotations

from dataclasses import dataclass
import random

from .env import ColorJumpEnv, GameConfig, State


WAIT = 0
TAP = 1


@dataclass
class TrainStats:
    episodes: int
    best_score: int
    average_score: float
    q_states: int
    eval_average_score: float | None = None
    eval_best_score: int | None = None


class QAgent:
    def __init__(
        self,
        q_table: dict[str, list[float]] | None = None,
        learning_rate: float = 0.14,
        discount: float = 0.92,
        epsilon: float = 0.18,
        seed: int | None = None,
    ):
        self.q_table = q_table or {}
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        self.min_epsilon = 0.02
        self.random = random.Random(seed)

    def choose_action(
        self,
        state: State,
        explore: bool = True,
        epsilon: float | None = None,
    ) -> int:
        current_epsilon = self.epsilon if epsilon is None else epsilon
        if explore and self.random.random() < current_epsilon:
            return self.random.choice([WAIT, TAP])

        wait_q, tap_q = self._values(state)
        if abs(tap_q - wait_q) < 2.0:
            return self._heuristic_action(state)

        return TAP if tap_q > wait_q else WAIT

    def learn(self, state: State, action: int, reward: float, next_state: State, done: bool) -> None:
        values = self._values(state)
        next_values = self._values(next_state)
        target = reward if done else reward + self.discount * max(next_values)
        values[action] += self.learning_rate * (target - values[action])
        self.q_table[state.key()] = values

    def train(
        self,
        episodes: int,
        seed: int | None = None,
        config: GameConfig | None = None,
        eval_episodes: int = 100,
    ) -> TrainStats:
        scores: list[int] = []
        best_score = 0

        for episode in range(episodes):
            progress = episode / max(1, episodes - 1)
            epsilon = self.epsilon + (self.min_epsilon - self.epsilon) * progress
            env_seed = None if seed is None else seed + episode
            env = ColorJumpEnv(config=config, seed=env_seed)
            state = env.reset()
            done = False

            while not done:
                action = self.choose_action(state, explore=True, epsilon=epsilon)
                result = env.step(action)
                self.learn(state, action, result.reward, result.state, result.done)
                state = result.state
                done = result.done

            scores.append(env.score)
            best_score = max(best_score, env.score)

        average = sum(scores) / len(scores) if scores else 0.0
        eval_stats = self.evaluate(
            eval_episodes,
            seed=None if seed is None else seed + episodes + 10_000,
            config=config,
        )
        return TrainStats(
            episodes=episodes,
            best_score=best_score,
            average_score=average,
            q_states=len(self.q_table),
            eval_average_score=eval_stats.average_score,
            eval_best_score=eval_stats.best_score,
        )

    def evaluate(
        self,
        episodes: int,
        seed: int | None = None,
        config: GameConfig | None = None,
    ) -> TrainStats:
        scores: list[int] = []
        best_score = 0

        for episode in range(episodes):
            env_seed = None if seed is None else seed + episode
            env = ColorJumpEnv(config=config, seed=env_seed)
            state = env.reset()
            done = False

            while not done:
                action = self.choose_action(state, explore=False)
                result = env.step(action)
                state = result.state
                done = result.done

            scores.append(env.score)
            best_score = max(best_score, env.score)

        average = sum(scores) / len(scores) if scores else 0.0
        return TrainStats(
            episodes=episodes,
            best_score=best_score,
            average_score=average,
            q_states=len(self.q_table),
        )

    def _values(self, state: State) -> list[float]:
        key = state.key()
        if key not in self.q_table:
            self.q_table[key] = [0.0, 0.0]
        return self.q_table[key]

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
