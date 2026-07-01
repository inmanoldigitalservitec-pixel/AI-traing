from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from .agent import QAgent, TAP
from .env import ColorJumpEnv, GameConfig
from .storage import load_q_table, reset_storage, save_q_table


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Color Jump AI MVP")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Entrena la IA")
    train.add_argument("--episodes", type=int, default=1000)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--max-steps", type=int, default=None)
    train.add_argument("--eval-episodes", type=int, default=100)

    evaluate = subparsers.add_parser("eval", help="Evalua la IA")
    evaluate.add_argument("--episodes", type=int, default=200)
    evaluate.add_argument("--seed", type=int, default=1000)
    evaluate.add_argument("--max-steps", type=int, default=None)

    play = subparsers.add_parser("play", help="Muestra una partida en texto")
    play.add_argument("--seed", type=int, default=7)
    play.add_argument("--max-steps", type=int, default=100000)

    export = subparsers.add_parser("export", help="Exporta policy JSON")
    export.add_argument("--output", type=Path, default=Path("policy.json"))

    subparsers.add_parser("reset", help="Borra la memoria entrenada")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "reset":
        removed = reset_storage()
        print("Memoria borrada." if removed else "No habia memoria guardada.")
        return

    q_table = load_q_table()
    agent = QAgent(q_table=q_table)

    if args.command == "train":
        config = build_game_config(args.max_steps)
        stats = agent.train(
            args.episodes,
            seed=args.seed,
            config=config,
            eval_episodes=args.eval_episodes,
        )
        save_q_table(agent.q_table)
        print(
            f"Entrenadas: {stats.episodes} | "
            f"Promedio: {stats.average_score:.2f} | "
            f"Mejor: {stats.best_score} | "
            f"Estados: {stats.q_states}"
        )
        print(
            f"Evaluacion post-train: "
            f"Promedio: {stats.eval_average_score:.2f} | "
            f"Mejor: {stats.eval_best_score}"
        )
        return

    if args.command == "eval":
        config = build_game_config(args.max_steps)
        stats = agent.evaluate(args.episodes, seed=args.seed, config=config)
        print(
            f"Evaluadas: {stats.episodes} | "
            f"Promedio: {stats.average_score:.2f} | "
            f"Mejor: {stats.best_score} | "
            f"Estados: {stats.q_states}"
        )
        return

    if args.command == "play":
        config = build_game_config(args.max_steps)
        play_episode(agent, seed=args.seed, config=config)
        return

    if args.command == "export":
        export_policy(agent, args.output)
        return


def build_game_config(max_steps: int | None) -> GameConfig:
    config = GameConfig()
    if max_steps is None:
        return config
    return replace(config, max_steps=max_steps)


def play_episode(agent: QAgent, seed: int, config: GameConfig) -> None:
    env = ColorJumpEnv(config=config, seed=seed)
    state = env.reset()
    done = False
    taps = 0
    waits = 0
    best_score = 0
    visited_states: set[str] = set()

    print("Partida IA")
    print("step | action | score | altitude | velocity | state")
    print("-" * 72)

    while not done:
        visited_states.add(state.key())
        action = agent.choose_action(state, explore=False)
        result = env.step(action)
        label = "tap" if action == TAP else "wait"
        if action == TAP:
            taps += 1
        else:
            waits += 1
        best_score = max(best_score, env.score)

        if env.steps % 8 == 0 or action == TAP or result.done:
            print(
                f"{env.steps:>4} | {label:<6} | {env.score:>5} | "
                f"{env.altitude:>8.1f} | {env.velocity:>8.1f} | {state.key()}"
            )

        state = result.state
        done = result.done

    print("-" * 72)
    print(f"Final: score={env.score}, reason={result.reason}")
    print("")
    print("Resumen")
    print("-" * 72)
    print(f"Score final: {env.score}")
    print(f"Mejor score durante la partida: {best_score}")
    print(f"Motivo de cierre: {result.reason}")
    print(f"Pasos totales: {env.steps}")
    print(f"Taps: {taps}")
    print(f"Waits: {waits}")
    print(f"Estados visitados: {len(visited_states)}")
    print(f"Altitud final: {env.altitude:.1f}")
    print(f"Velocidad final: {env.velocity:.1f}")


def export_policy(agent: QAgent, output: Path) -> None:
    policy = {
        key: "tap" if values[1] > values[0] else "wait"
        for key, values in agent.q_table.items()
    }
    output.write_text(json.dumps(policy, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Politica exportada: {output} ({len(policy)} estados)")
