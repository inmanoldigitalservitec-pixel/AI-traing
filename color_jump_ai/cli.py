from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agent import QAgent, TAP
from .env import ColorJumpEnv
from .storage import load_q_table, reset_storage, save_q_table


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Color Jump AI MVP")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Entrena la IA")
    train.add_argument("--episodes", type=int, default=1000)
    train.add_argument("--seed", type=int, default=42)

    evaluate = subparsers.add_parser("eval", help="Evalua la IA")
    evaluate.add_argument("--episodes", type=int, default=200)
    evaluate.add_argument("--seed", type=int, default=1000)

    play = subparsers.add_parser("play", help="Muestra una partida en texto")
    play.add_argument("--seed", type=int, default=7)

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
        stats = agent.train(args.episodes, seed=args.seed)
        save_q_table(agent.q_table)
        print(
            f"Entrenadas: {stats.episodes} | "
            f"Promedio: {stats.average_score:.2f} | "
            f"Mejor: {stats.best_score} | "
            f"Estados: {stats.q_states}"
        )
        return

    if args.command == "eval":
        stats = agent.evaluate(args.episodes, seed=args.seed)
        print(
            f"Evaluadas: {stats.episodes} | "
            f"Promedio: {stats.average_score:.2f} | "
            f"Mejor: {stats.best_score} | "
            f"Estados: {stats.q_states}"
        )
        return

    if args.command == "play":
        play_episode(agent, seed=args.seed)
        return

    if args.command == "export":
        export_policy(agent, args.output)
        return


def play_episode(agent: QAgent, seed: int) -> None:
    env = ColorJumpEnv(seed=seed)
    state = env.reset()
    done = False

    print("Partida IA")
    print("step | action | score | altitude | velocity | state")
    print("-" * 72)

    while not done:
        action = agent.choose_action(state, explore=False)
        result = env.step(action)
        label = "tap" if action == TAP else "wait"

        if env.steps % 8 == 0 or action == TAP or result.done:
            print(
                f"{env.steps:>4} | {label:<6} | {env.score:>5} | "
                f"{env.altitude:>8.1f} | {env.velocity:>8.1f} | {state.key()}"
            )

        state = result.state
        done = result.done

    print("-" * 72)
    print(f"Final: score={env.score}, reason={result.reason}")


def export_policy(agent: QAgent, output: Path) -> None:
    policy = {
        key: "tap" if values[1] > values[0] else "wait"
        for key, values in agent.q_table.items()
    }
    output.write_text(json.dumps(policy, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Politica exportada: {output} ({len(policy)} estados)")
