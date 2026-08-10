import unittest

import numpy as np

from autoplay.domain import Direction, Move, apply_move, completed_lines, legal_moves
from autoplay.solver import Solver


class DomainTests(unittest.TestCase):
    def test_row_wraps_left(self):
        board = np.array([[1, 2, 3], [4, 5, 1]])
        expected = np.array([[2, 3, 1], [4, 5, 1]])
        np.testing.assert_array_equal(apply_move(board, Move(0, 0, Direction.LEFT)), expected)

    def test_column_wraps_down(self):
        board = np.array([[1, 2], [3, 4], [5, 1]])
        expected = np.array([[5, 2], [1, 4], [3, 1]])
        np.testing.assert_array_equal(apply_move(board, Move(0, 0, Direction.DOWN)), expected)

    def test_legal_move_count(self):
        self.assertEqual(len(list(legal_moves(np.ones((4, 5))))), 18)

    def test_yoshi_is_wildcard_in_complete_line(self):
        rows, cols = completed_lines(np.array([[2, 5, 2], [1, 3, 4], [3, 4, 1]]))
        self.assertEqual(rows, (0,))
        self.assertEqual(cols, ())

    def test_solver_prioritizes_immediate_clear(self):
        board = np.array([[1, 2, 1], [2, 1, 2], [3, 1, 1]])
        move = Solver().best_move(board)
        shifted = apply_move(board, move)
        self.assertTrue(any(completed_lines(shifted)))


if __name__ == "__main__":
    unittest.main()
