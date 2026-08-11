"""Tipos y reglas puras de Yoshi's Cookie, sin dependencias gráficas."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Iterable, Tuple

import numpy as np


class Direction(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    UP = "up"
    DOWN = "down"


class CookieType(IntEnum):
    UNKNOWN = 0
    DIAMOND = 1
    HEART = 2
    FLOWER = 3
    CHECKER = 4
    CIRCLE = 5
    YOSHI = 6


@dataclass(frozen=True)
class Move:
    """Desplaza cíclicamente una fila o columna una posición."""

    row: int
    col: int
    direction: Direction
    score: float = 0.0

    @property
    def axis_index(self) -> int:
        return self.row if self.direction in (Direction.LEFT, Direction.RIGHT) else self.col


def validate_board(board: np.ndarray) -> np.ndarray:
    result = np.asarray(board, dtype=np.int8)
    if result.ndim != 2 or not result.size:
        raise ValueError("El tablero debe ser una matriz 2D no vacía")
    if np.any((result < CookieType.UNKNOWN) | (result > CookieType.YOSHI)):
        raise ValueError("El tablero contiene un tipo de cookie inválido")
    return result


def apply_move(board: np.ndarray, move: Move) -> np.ndarray:
    """Aplica la rotación toroidal usada por el juego."""
    result = validate_board(board).copy()
    rows, cols = result.shape
    if not (0 <= move.row < rows and 0 <= move.col < cols):
        raise ValueError(f"Cursor fuera del tablero: ({move.row}, {move.col})")
    if move.direction == Direction.LEFT:
        result[move.row] = np.roll(result[move.row], -1)
    elif move.direction == Direction.RIGHT:
        result[move.row] = np.roll(result[move.row], 1)
    elif move.direction == Direction.UP:
        result[:, move.col] = np.roll(result[:, move.col], -1)
    elif move.direction == Direction.DOWN:
        result[:, move.col] = np.roll(result[:, move.col], 1)
    return result


def legal_moves(board: np.ndarray) -> Iterable[Move]:
    rows, cols = validate_board(board).shape
    for row in range(rows):
        yield Move(row, 0, Direction.LEFT)
        yield Move(row, 0, Direction.RIGHT)
    for col in range(cols):
        yield Move(0, col, Direction.UP)
        yield Move(0, col, Direction.DOWN)


def completed_lines(board: np.ndarray) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Devuelve líneas completas; 6 es Yoshi y 0 nunca forma coincidencias."""
    grid = validate_board(board)

    def is_match(line: np.ndarray) -> bool:
        values = line[(line != CookieType.UNKNOWN) & (line != CookieType.YOSHI)]
        return bool(np.all(line != 0) and (not values.size or np.all(values == values[0])))

    matched_rows = tuple(i for i, row in enumerate(grid) if is_match(row))
    matched_cols = tuple(i for i, col in enumerate(grid.T) if is_match(col))
    return matched_rows, matched_cols


def matched_cell_count(board: np.ndarray) -> int:
    rows, cols = completed_lines(board)
    cells = {(row, col) for row in rows for col in range(board.shape[1])}
    cells.update((row, col) for col in cols for row in range(board.shape[0]))
    return len(cells)
