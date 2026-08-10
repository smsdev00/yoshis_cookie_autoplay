import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from unittest.mock import patch

from autoplay.bsnes import BSNES_KEYS, BsnesController, BsnesNativeDetector, BsnesScreenshotSource
from autoplay.domain import Direction, Move


class FakeInput:
    def __init__(self, callback=None):
        self.callback = callback
        self.buttons = []

    def tap(self, button):
        self.buttons.append(button)
        if self.callback:
            self.callback()


def cookie_frame(values):
    detector = BsnesNativeDetector()
    frame = np.zeros((224, 256, 3), dtype=np.uint8)
    centers = {
        1: (8, 255, 0),
        2: (0, 0, 198),
        3: (0, 66, 255),
        4: (0, 255, 255),
        5: (255, 0, 255),
    }
    height, width = values.shape
    top = detector.BOTTOM - (height - 1) * detector.CELL
    for row in range(height):
        for col in range(width):
            x = detector.LEFT + col * detector.CELL
            y = top + row * detector.CELL
            frame[y - 7:y + 8, x - 7:x + 8] = (100, 200, 240)
            frame[y, x] = centers[int(values[row, col])]
    return frame


class BsnesDetectorTests(unittest.TestCase):
    def test_native_frame_to_board(self):
        expected = np.array([[3, 4, 3], [1, 3, 1], [2, 1, 2]], dtype=np.int8)
        observation = BsnesNativeDetector().detect(cookie_frame(expected))
        np.testing.assert_array_equal(observation.board, expected)

    def test_rejects_wrong_resolution(self):
        with self.assertRaisesRegex(ValueError, "256x224"):
            BsnesNativeDetector().detect(np.zeros((100, 100, 3), dtype=np.uint8))

    def test_screenshot_source_waits_for_new_complete_bmp(self):
        with TemporaryDirectory() as directory:
            target = Path(directory) / "Yoshi's Cookie (USA)-001.bmp"

            def write_later():
                def write():
                    time.sleep(0.03)
                    cv2.imwrite(str(target), np.zeros((224, 256, 3), dtype=np.uint8))
                threading.Thread(target=write, daemon=True).start()

            fake = FakeInput(write_later)
            source = BsnesScreenshotSource(fake, Path(directory), timeout=1.0, poll_interval=0.01)
            image = source.capture()
            self.assertEqual(fake.buttons, ["screenshot"])
            self.assertEqual(image.shape, (224, 256, 3))

    @patch("autoplay.adapters.shutil.which", return_value="/usr/bin/ydotool")
    @patch("autoplay.adapters.subprocess.run")
    def test_cursor_moves_with_shifted_cookie(self, run, _which):
        run.return_value.returncode = 0
        run.return_value.stderr = ""
        controller = BsnesController(BSNES_KEYS)
        controller.tap = lambda _button: None
        cursor = controller.execute(Move(0, 0, Direction.LEFT), (0, 0), (3, 4))
        self.assertEqual(cursor, (0, 3))


if __name__ == "__main__":
    unittest.main()
