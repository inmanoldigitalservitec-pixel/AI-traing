from __future__ import annotations

import json
from pathlib import Path


APP_DIR = Path(".color_jump_ai")
Q_TABLE_PATH = APP_DIR / "q_table.json"
DQN_MODEL_PATH = APP_DIR / "dqn_model.pt"


def load_q_table(path: Path = Q_TABLE_PATH) -> dict[str, list[float]]:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    return {str(key): [float(value[0]), float(value[1])] for key, value in data.items()}


def save_q_table(q_table: dict[str, list[float]], path: Path = Q_TABLE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(q_table, file, indent=2, sort_keys=True)


def reset_storage(path: Path = Q_TABLE_PATH) -> bool:
    if not path.exists():
        return False
    path.unlink()
    return True
