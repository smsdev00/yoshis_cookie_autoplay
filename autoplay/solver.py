"""Búsqueda determinista de movimientos para tableros pequeños."""

from __future__ import annotations

from dataclasses import replace
from typing import List, Optional

import numpy as np

from autoplay.domain import (CookieType, Move, apply_move, completed_lines,
                             legal_moves, matched_cell_count, validate_board)


class Solver:
    def rank(self, board: np.ndarray) -> List[Move]:
        grid = validate_board(board)
        if np.any(grid == CookieType.UNKNOWN):
            raise ValueError("No se puede resolver un tablero con cookies desconocidas")
        ranked = []
        for move in legal_moves(grid):
            candidate = apply_move(grid, move)
            rows, cols = completed_lines(candidate)
            cleared = matched_cell_count(candidate)
            # Prioridad absoluta a limpiar. La cohesión sirve para desempatar y
            # para proponer un setup cuando aún no existe una limpieza inmediata.
            score = cleared * 1000 + (len(rows) + len(cols)) * 100 + self._cohesion(candidate)
            ranked.append(replace(move, score=float(score)))
        return sorted(ranked, key=lambda item: (-item.score, item.direction.value, item.axis_index))

    def best_move(self, board: np.ndarray) -> Optional[Move]:
        moves = self.rank(board)
        return moves[0] if moves else None

    @staticmethod
    def _cohesion(board: np.ndarray) -> int:
        def line_score(line: np.ndarray) -> int:
            normal = line[(line != CookieType.UNKNOWN) & (line != CookieType.YOSHI)]
            if not normal.size:
                return 0
            counts = np.bincount(normal)
            return int(counts.max() ** 2 +
                       np.count_nonzero(line == CookieType.YOSHI) * counts.max())

        return sum(line_score(row) for row in board) + sum(line_score(col) for col in board.T)
