"""Ciclo captura → estabilidad → solución → ejecución → verificación."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

from autoplay.domain import Move, apply_move
from autoplay.solver import Solver


class PostMoveVerificationError(RuntimeError):
    """El juego recibió un movimiento pero su resultado no pudo validarse."""


@dataclass
class CycleConfig:
    poll_interval: float = 0.25
    stable_frames: int = 3
    animation_delay: float = 0.8
    stability_timeout: float = 3.0
    clear_stability_timeout: float = 40.0
    min_confidence: float = 0.98
    require_cursor: bool = False
    cursor_settle_delay: float = 0.2


@dataclass
class Observation:
    board: np.ndarray
    confidence: float
    cursor: Optional[Tuple[int, int]] = None


class AutoPlayLoop:
    def __init__(self, observe: Callable[[], Observation], solver: Optional[Solver] = None,
                 executor=None, config: Optional[CycleConfig] = None):
        self.observe = observe
        self.solver = solver or Solver()
        self.executor = executor
        self.config = config or CycleConfig()
        self.cursor: Tuple[int, int] = (0, 0)

    def wait_for_stable_board(self, timeout: Optional[float] = None) -> Observation:
        deadline = time.monotonic() + (
            self.config.stability_timeout if timeout is None else timeout
        )
        previous = None
        consecutive = 0
        latest = None
        last_error = None
        stable_cursor = None
        while time.monotonic() < deadline:
            try:
                latest = self.observe()
            except (RuntimeError, ValueError) as exc:
                # Menús y animaciones producen temporalmente frames sin tablero.
                last_error = exc
                previous, consecutive = None, 0
                stable_cursor = None
                time.sleep(self.config.poll_interval)
                continue
            if latest.confidence < self.config.min_confidence:
                previous, consecutive = None, 0
                stable_cursor = None
            elif previous is not None and np.array_equal(previous, latest.board):
                consecutive += 1
            else:
                previous, consecutive = latest.board.copy(), 1
                stable_cursor = None
            if latest.cursor is not None:
                stable_cursor = latest.cursor
            if consecutive >= self.config.stable_frames and (
                not self.config.require_cursor or stable_cursor is not None
            ):
                return Observation(latest.board, latest.confidence, stable_cursor)
            time.sleep(self.config.poll_interval)
        detail = f": {last_error}" if last_error else ""
        raise TimeoutError(f"El tablero no se estabilizó antes del timeout{detail}")

    def propose(self) -> Tuple[Observation, Move]:
        observation = self.wait_for_stable_board()
        if observation.cursor is not None:
            self.cursor = observation.cursor
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
        self.position_cursor((move.row, move.col), before.board.shape)
        self.cursor = self.executor.execute(move, self.cursor, before.board.shape)
        time.sleep(self.config.animation_delay)
        expected = apply_move(before.board, move)
        from autoplay.domain import completed_lines
        clears_line = any(completed_lines(expected))
        try:
            after = self.wait_for_stable_board(
                self.config.clear_stability_timeout if clears_line else None
            )
        except Exception as exc:
            raise PostMoveVerificationError(
                f"No se pudo observar el resultado del movimiento: {exc}"
            ) from exc
        # La limpieza/entrada de cookies puede cambiar dimensiones o contenido.
        # Si no hubo línea inmediata, el desplazamiento sí debe coincidir exacto.
        if not clears_line and not np.array_equal(expected, after.board):
            raise PostMoveVerificationError(
                "La verificación falló: el tablero observado no coincide con el movimiento"
            )
        return before, move, after

    def position_cursor(self, target: Tuple[int, int],
                        board_shape: Tuple[int, int]) -> Tuple[int, int]:
        """Navega una tecla por vez y confirma visualmente cada nueva posición."""
        if self.executor is None:
            raise RuntimeError("No hay backend de entrada configurado")
        rows, cols = board_shape
        max_steps = (rows + cols) * 3
        for _ in range(max_steps):
            if self.cursor == target:
                return self.cursor
            row, col = self.cursor
            target_row, target_col = target
            if row < target_row:
                direction = "down"
            elif row > target_row:
                direction = "up"
            elif col < target_col:
                direction = "right"
            else:
                direction = "left"
            self.executor.step_cursor(direction)
            time.sleep(self.config.cursor_settle_delay)
            observed = self.wait_for_stable_board()
            if observed.board.shape != board_shape or observed.cursor is None:
                raise RuntimeError("No se pudo verificar el cursor durante la navegación")
            self.cursor = observed.cursor
        raise RuntimeError(f"El cursor no llegó a {target}; observado {self.cursor}")
