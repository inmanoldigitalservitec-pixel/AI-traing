# Color Jump AI MVP

MVP en Python para entrenar una IA sencilla que aprenda a jugar una version simulada de Color Jump.

La IA usa Q-learning con dos acciones:

- `tap`: saltar
- `wait`: esperar

El entrenamiento corre sin graficos para poder simular muchas partidas rapido. El repo incluye dos cerebros:

- Q-learning con tabla: simple, rapido y sin dependencias.
- DQN con PyTorch: red neuronal para el siguiente nivel.

## Requisitos

- Python 3.10+
- Q-learning no requiere dependencias externas
- DQN requiere PyTorch

Instalar PyTorch:

```bash
python3 -m pip install torch
```

## Flujo recomendado

Este es el flujo principal para entrenar, medir y ver la IA jugando:

```bash
python3 -m color_jump_ai train --episodes 10000 --max-steps 10000
python3 -m color_jump_ai eval --episodes 100 --max-steps 10000
python3 -m color_jump_ai play --max-steps 50000
```

Si actualizaste desde una version anterior, borra la memoria vieja antes de entrenar:

```bash
python3 -m color_jump_ai reset
```

## Flujo DQN con PyTorch

Entrenar la red neuronal:

```bash
python3 -m color_jump_ai dqn-train --episodes 3000 --max-steps 10000
```

Evaluar la red neuronal:

```bash
python3 -m color_jump_ai dqn-eval --episodes 100 --max-steps 10000
```

Ver la red neuronal jugando:

```bash
python3 -m color_jump_ai dqn-play --max-steps 50000
```

El modelo DQN se guarda en:

```text
.color_jump_ai/dqn_model.pt
```

## Comandos

Entrenar 5000 partidas:

```bash
python -m color_jump_ai train --episodes 5000
```

Evaluar la IA entrenada:

```bash
python -m color_jump_ai eval --episodes 200
```

Ver una partida en modo texto:

```bash
python -m color_jump_ai play
```

Por defecto `play` corre hasta `100000` pasos para que puedas observar una partida larga. Puedes ajustar la duracion:

```bash
python -m color_jump_ai play --max-steps 10000
python -m color_jump_ai play --max-steps 250000
```

Tambien puedes evaluar o entrenar con mas pasos por partida:

```bash
python -m color_jump_ai eval --episodes 50 --max-steps 10000
python -m color_jump_ai train --episodes 5000 --max-steps 10000
```

Exportar la politica para usarla luego en el juego web:

```bash
python -m color_jump_ai export --output policy.json
```

Reiniciar memoria:

```bash
python -m color_jump_ai reset
```

## Archivos generados

- `.color_jump_ai/q_table.json`: memoria aprendida por la IA.
- `policy.json`: politica exportada para integrarla con JavaScript.
