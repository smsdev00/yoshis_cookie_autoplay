import unittest
from pathlib import Path

import numpy as np

from config import CONF
from main import ImprovedCookieDetector


ROOT = Path(__file__).resolve().parents[1]


class DetectionRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector = ImprovedCookieDetector(CONF)

    def detect(self, filename):
        path = ROOT / "imgs" / filename
        cookies = self.detector.detectar_cookies(str(path))
        grid, info = self.detector.construir_grilla_inteligente(cookies)
        self.assertEqual(info["colisiones"], 0)
        self.assertEqual(info["confianza"], 1.0)
        return grid

    def test_round_1_full_board(self):
        expected = np.array(
            [
                [2, 3, 2, 3],
                [3, 2, 3, 2],
                [2, 3, 2, 3],
                [3, 2, 3, 2],
            ]
        )
        np.testing.assert_array_equal(self.detect("R01S01.png"), expected)

    def test_round_1_selects_lower_left_component(self):
        np.testing.assert_array_equal(self.detect("R01S02.png"), np.array([[3, 2], [2, 3]]))
        np.testing.assert_array_equal(self.detect("R01S03.png"), np.array([[3], [2]]))

    def test_round_2_ignores_falling_row(self):
        expected = np.array(
            [
                [3, 2, 3, 1],
                [1, 3, 1, 2],
                [2, 1, 2, 1],
                [3, 2, 3, 2],
            ]
        )
        np.testing.assert_array_equal(self.detect("R02S01.png"), expected)

    def test_round_2_sparse_lower_left_component(self):
        np.testing.assert_array_equal(self.detect("R02S02.png"), np.array([[3, 3, 1]]))

    def test_round_2_classifies_square_under_crosshair(self):
        expected = np.array(
            [
                [4, 2, 1, 1],
                [4, 2, 1, 4],
                [3, 3, 4, 1],
            ]
        )
        np.testing.assert_array_equal(self.detect("R02S03.png"), expected)


if __name__ == "__main__":
    unittest.main()
