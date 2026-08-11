import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from autoplay.domain import Direction, Move
from autoplay.hud import HudDisplay, KioskTelemetry, StatsStore


class StatsStoreTests(unittest.TestCase):
    def test_persists_cumulative_history(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "stats.json"
            store = StatsStore(path)
            store.begin_session()
            store.record_move(1, 52)
            store.record_move(2, 16480)
            store.record_stage(3)

            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["sessions"], 1)
            self.assertEqual(saved["total_moves"], 2)
            self.assertEqual(saved["longest_session_moves"], 2)
            self.assertEqual(saved["max_stage"], 3)
            self.assertEqual(saved["max_tactical_score"], 16480)
            self.assertFalse(path.with_suffix(".json.tmp").exists())

    def test_telemetry_updates_live_snapshot_without_a_display(self):
        with TemporaryDirectory() as directory:
            telemetry = KioskTelemetry(Path(directory) / "stats.json")
            telemetry.start()
            telemetry.stage_start()
            telemetry.move(
                1, np.array([[1, 2], [3, 6]]),
                Move(0, 0, Direction.DOWN, score=1234),
            )
            self.assertEqual(telemetry.snapshot.stage, 2)
            self.assertEqual(telemetry.snapshot.moves, 1)
            self.assertEqual(telemetry.snapshot.board, [[1, 2], [3, 6]])
            self.assertEqual(telemetry.snapshot.last_move, "DOWN / AXIS 0")


class HudFormattingTests(unittest.TestCase):
    def test_board_uses_compact_cookie_glyphs(self):
        self.assertEqual(
            HudDisplay._board_text([[1, 2, 3], [4, 5, 6]]),
            "D H F\nX O Y",
        )


if __name__ == "__main__":
    unittest.main()
