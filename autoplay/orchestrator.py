"""Ciclo captura → estabilidad → solución → ejecución → verificación."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

from autoplay.domain import Move, apply_move
from autoplay.solver import Solver


@dataclass
class CycleConfig:
    poll_interval: float = 0.25
    stable_frames: int = 3
    animation_delay: float = 0.8
    stability_timeout: float = 3.0
    min_confidence: float = 0.98


@dataclass
class Observation:
    board: np.ndarray
    confidence: float


class AutoPlayLoop:
    def __init__(self, observe: Callable[[], Observation], solver: Optional[Solver] = None,
                 executor=None, config: Optional[CycleConfig] = None):
        self.observe = observe
        self.solver = solver or Solver()
        self.executor = executor
        self.config = config or CycleConfig()
        self.cursor: Tuple[int, int] = (0, 0)

    def wait_for_stable_board(self) -> Observation:
        deadline = time.monotonic() + self.config.stability_timeout
        previous = None
        consecutive = 0
        latest = None
        while time.monotonic() < deadline:
            latest = self.observe()
            if latest.confidence < self.config.min_confidence:
                previous, consecutive = None, 0
            elif previous is not None and np.array_equal(previous, latest.board):
                consecutive += 1
            else:
                previous, consecutive = latest.board.copy(), 1
            if consecutive >= self.config.stable_frames:
                return latest
            time.sleep(self.config.poll_interval)
        raise TimeoutError("El tablero no se estabilizó antes del timeout")

    def propose(self) -> Tuple[Observation, Move]:
        observation = self.wait_for_stable_board()
        move = self.solver.best_move(observation.board)
        if move is None:
            raise RuntimeError("No hay movimientos legales")
        return observation, move

    def step(self, execute: bool = False) -> Tuple[Observation, Move, Optional[Observation]]:
        before, move = self.propose()
        if not execute:
            return before, move, None
        if self.executor is None:
            raise RuntimeError("No hay backend de entrada configurado")
        self.cursor = self.executor.execute(move, self.cursor, before.board.shape)
        time.sleep(self.config.animation_delay)
        after = self.wait_for_stable_board()
        expected = apply_move(before.board, move)
        # La limpieza/entrada de cookies puede cambiar dimensiones o contenido.
        # Si no hubo línea inmediata, el desplazamiento sí debe coincidir exacto.
        from autoplay.domain import completed_lines
        if not any(completed_lines(expected)) and not np.array_equal(expected, after.board):
            raise RuntimeError("La verificación falló: el tablero observado no coincide con el movimiento")
        return before, move, after
