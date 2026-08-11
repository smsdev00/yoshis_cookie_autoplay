import unittest
from unittest.mock import patch

import numpy as np

from autoplay.orchestrator import (AutoPlayLoop, CycleConfig, Observation,
                                   matches_expected_with_growth)


class OrchestratorTests(unittest.TestCase):
    def test_verification_accepts_growth_only_at_top_and_right(self):
        expected = np.array([[1, 2], [3, 4]])
        observed = np.array([[5, 5, 5], [1, 2, 6], [3, 4, 6]])
        self.assertTrue(matches_expected_with_growth(expected, observed))
        self.assertFalse(matches_expected_with_growth(expected, observed[:, 1:]))

    @patch("autoplay.orchestrator.time.sleep", return_value=None)
    def test_requires_consecutive_equal_boards(self, _sleep):
        changing = np.array([[1, 2], [2, 1]])
        stable = np.array([[2, 1], [1, 2]])
        observations = iter([
            Observation(changing, 1.0), Observation(stable, 1.0),
            Observation(stable, 1.0), Observation(stable, 1.0),
        ])
        loop = AutoPlayLoop(lambda: next(observations), config=CycleConfig(stable_frames=3))
        np.testing.assert_array_equal(loop.wait_for_stable_board().board, stable)

    @patch("autoplay.orchestrator.time.sleep", return_value=None)
    def test_low_confidence_resets_stability(self, _sleep):
        board = np.array([[1, 2], [2, 1]])
        observations = iter([
            Observation(board, 1.0), Observation(board, 0.5),
            Observation(board, 1.0), Observation(board, 1.0),
        ])
        loop = AutoPlayLoop(lambda: next(observations),
                            config=CycleConfig(stable_frames=2, min_confidence=0.9))
        self.assertEqual(loop.wait_for_stable_board().confidence, 1.0)

    @patch("autoplay.orchestrator.time.sleep", return_value=None)
    def test_retries_frames_without_a_playable_board(self, _sleep):
        board = np.array([[1, 2], [2, 1]])
        observations = iter([
            ValueError("Tablero no jugable o en animación: 0x0"),
            Observation(board, 1.0), Observation(board, 1.0),
        ])

        def observe():
            result = next(observations)
            if isinstance(result, Exception):
                raise result
            return result

        loop = AutoPlayLoop(observe, config=CycleConfig(stable_frames=2))
        np.testing.assert_array_equal(loop.wait_for_stable_board().board, board)

    @patch("autoplay.orchestrator.time.sleep", return_value=None)
    def test_propose_synchronizes_detected_cursor(self, _sleep):
        board = np.array([[1, 2], [2, 1]])
        observation = Observation(board, 1.0, cursor=(1, 1))
        loop = AutoPlayLoop(
            lambda: observation, config=CycleConfig(stable_frames=2)
        )
        loop.propose()
        self.assertEqual(loop.cursor, (1, 1))

    @patch("autoplay.orchestrator.time.sleep", return_value=None)
    def test_stable_board_retains_cursor_seen_during_blink(self, _sleep):
        board = np.array([[1, 2], [2, 1]])
        observations = iter([
            Observation(board, 1.0, cursor=(1, 1)),
            Observation(board, 1.0, cursor=None),
        ])
        loop = AutoPlayLoop(
            lambda: next(observations),
            config=CycleConfig(stable_frames=2, require_cursor=True),
        )
        result = loop.wait_for_stable_board()
        self.assertEqual(result.cursor, (1, 1))

    @patch("autoplay.orchestrator.time.sleep", return_value=None)
    def test_required_cursor_waits_past_stable_frames_until_visible(self, _sleep):
        board = np.array([[1, 2], [2, 1]])
        observations = iter([
            Observation(board, 1.0), Observation(board, 1.0),
            Observation(board, 1.0, cursor=(0, 1)),
        ])
        loop = AutoPlayLoop(
            lambda: next(observations),
            config=CycleConfig(stable_frames=2, require_cursor=True),
        )
        result = loop.wait_for_stable_board()
        self.assertEqual(result.cursor, (0, 1))

    @patch("autoplay.orchestrator.time.sleep", return_value=None)
    def test_closed_loop_navigation_reobserves_each_cursor_step(self, _sleep):
        board = np.array([[1, 2, 1], [2, 1, 2]])

        class Executor:
            def __init__(self):
                self.directions = []

            def step_cursor(self, direction):
                self.directions.append(direction)

        executor = Executor()
        observations = iter([
            Observation(board, 1.0, cursor=(0, 2)),
            Observation(board, 1.0, cursor=(0, 2)),
            Observation(board, 1.0, cursor=(0, 0)),
            Observation(board, 1.0, cursor=(0, 0)),
        ])
        loop = AutoPlayLoop(
            lambda: next(observations), executor=executor,
            config=CycleConfig(stable_frames=2, require_cursor=True),
        )
        loop.cursor = (1, 2)
        self.assertEqual(loop.position_cursor((0, 0), board.shape), (0, 0))
        self.assertEqual(executor.directions, ["up", "left"])


if __name__ == "__main__":
    unittest.main()
