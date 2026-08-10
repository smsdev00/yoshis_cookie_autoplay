"""Compatibilidad con imports antiguos; usar :class:`autoplay.solver.Solver`."""

from autoplay.solver import Solver


class CookieMovementAnalyzer(Solver):
    def analyze_optimal_move(self, grid, strategy="balanced"):
        moves = self.rank(grid)
        return {"best_move": moves[0] if moves else None, "all_moves": moves}
