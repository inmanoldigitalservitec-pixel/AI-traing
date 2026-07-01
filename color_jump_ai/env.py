from __future__ import annotations

from dataclasses import dataclass
import math
import random


@dataclass(frozen=True)
class GameConfig:
    jump_force: float = 9.5
    gravity: float = 0.30
    fall_limit: float = -70.0
    fall_grace_frames: int = 62
    first_obstacle_altitude: float = 300.0
    obstacle_spacing: float = 380.0
    obstacle_radius: float = 72.0
    pass_padding: float = 35.0
    color_count: int = 4
    max_steps: int = 3500
    min_rotation_speed: float = 0.004
    max_rotation_speed: float = 0.010
    tap_penalty: float = 0.04
    tap_cooldown_frames: int = 6


@dataclass
class Obstacle:
    altitude: float
    color_offset: int
    rotation: float
    speed: float
    passed: bool = False


@dataclass(frozen=True)
class State:
    distance_bucket: int
    velocity_bucket: int
    ball_color: int
    target_color: int
    falling: int
    tap_ready: int

    def key(self) -> str:
        return (
            f"{self.distance_bucket}|{self.velocity_bucket}|"
            f"{self.ball_color}|{self.target_color}|{self.falling}|{self.tap_ready}"
        )


@dataclass
class StepResult:
    state: State
    reward: float
    done: bool
    score: int
    reason: str | None = None


class ColorJumpEnv:
    def __init__(self, config: GameConfig | None = None, seed: int | None = None):
        self.config = config or GameConfig()
        self.random = random.Random(seed)
        self.altitude = 0.0
        self.velocity = 0.0
        self.fall_frames = 0
        self.ball_color = 0
        self.score = 0
        self.steps = 0
        self.tap_cooldown = 0
        self.next_obstacle: Obstacle | None = None
        self.reset()

    def reset(self) -> State:
        self.altitude = 0.0
        self.velocity = 0.0
        self.fall_frames = 0
        self.ball_color = self.random.randrange(self.config.color_count)
        self.score = 0
        self.steps = 0
        self.tap_cooldown = 0
        self.next_obstacle = self._make_obstacle(self.config.first_obstacle_altitude)
        return self.state()

    def step(self, action: int) -> StepResult:
        # action 0 = wait, action 1 = tap
        tap_accepted = action == 1 and self.tap_cooldown == 0
        if tap_accepted:
            self.velocity = self.config.jump_force
            self.fall_frames = 0
            self.tap_cooldown = self.config.tap_cooldown_frames

        self.velocity -= self.config.gravity
        self.altitude += self.velocity
        self.steps += 1
        if self.tap_cooldown > 0:
            self.tap_cooldown -= 1

        reward = 0.02
        if tap_accepted:
            reward -= self.config.tap_penalty
        elif action == 1:
            reward -= self.config.tap_penalty * 2
        done = False
        reason = None

        if self.altitude < self.config.fall_limit and self.velocity < 0:
            self.fall_frames += 1
        else:
            self.fall_frames = 0

        if self.fall_frames >= self.config.fall_grace_frames:
            return StepResult(self.state(), -8.0, True, self.score, "fell")

        obstacle = self._current_obstacle()
        obstacle.rotation += obstacle.speed
        distance = obstacle.altitude - self.altitude

        touched = abs(distance) <= self.config.obstacle_radius
        if touched:
            target_color = self._target_color(obstacle)
            if target_color != self.ball_color:
                return StepResult(self.state(), -10.0, True, self.score, "wrong_color")
            reward += 0.08

        pass_line = obstacle.altitude + self.config.obstacle_radius + self.config.pass_padding
        if not obstacle.passed and self.altitude > pass_line:
            obstacle.passed = True
            self.score += 1
            reward += 10.0
            self.ball_color = self._next_ball_color()
            self.next_obstacle = self._make_obstacle(
                obstacle.altitude + self.config.obstacle_spacing
            )

        if self.steps >= self.config.max_steps:
            done = True
            reason = "max_steps"

        return StepResult(self.state(), reward, done, self.score, reason)

    def state(self) -> State:
        obstacle = self._current_obstacle()
        distance = obstacle.altitude - self.altitude
        target_color = self._target_color(obstacle)
        return State(
            distance_bucket=int(round(distance / 25.0) * 25),
            velocity_bucket=int(round(self.velocity / 1.5) * 1.5),
            ball_color=self.ball_color,
            target_color=target_color,
            falling=1 if self.velocity < 0 else 0,
            tap_ready=1 if self.tap_cooldown == 0 else 0,
        )

    def _current_obstacle(self) -> Obstacle:
        if self.next_obstacle is None:
            self.next_obstacle = self._make_obstacle(self.config.first_obstacle_altitude)
        return self.next_obstacle

    def _make_obstacle(self, altitude: float) -> Obstacle:
        return Obstacle(
            altitude=altitude,
            color_offset=self.random.randrange(self.config.color_count),
            rotation=self.random.random() * math.tau,
            speed=self.random.uniform(
                self.config.min_rotation_speed,
                self.config.max_rotation_speed,
            ),
        )

    def _target_color(self, obstacle: Obstacle) -> int:
        # Approximation of the segment at the vertical crossing line.
        quadrant = int(((math.pi / 2 - obstacle.rotation) % math.tau) / (math.pi / 2))
        return (quadrant + obstacle.color_offset) % self.config.color_count

    def _next_ball_color(self) -> int:
        next_color = self.ball_color
        while next_color == self.ball_color:
            next_color = self.random.randrange(self.config.color_count)
        return next_color
