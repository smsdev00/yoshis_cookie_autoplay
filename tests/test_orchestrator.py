import unittest
from unittest.mock import patch

import numpy as np

from autoplay.orchestrator import AutoPlayLoop, CycleConfig, Observation


class OrchestratorTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
