# Color Jump AI MVP

MVP en Python para entrenar una IA sencilla que aprenda a jugar una version simulada de Color Jump.

La IA usa Q-learning con dos acciones:

- `tap`: saltar
- `wait`: esperar

El entrenamiento corre sin graficos para poder simular muchas partidas rapido.

## Requisitos

- Python 3.10+
- No requiere dependencias externas

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
