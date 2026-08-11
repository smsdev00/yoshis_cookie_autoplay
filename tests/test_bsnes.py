import threading
import time
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from unittest.mock import Mock, patch

from autoplay.adapters import PersistentUInputBackend, YdotoolInputBackend
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
        5: (255, 100, 100),
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
    @patch("autoplay.bsnes.time.sleep", return_value=None)
    def test_prepare_uses_the_verified_startup_sequence(self, _sleep):
        backend = FakeInput()
        BsnesController(backend).prepare(
            launch_delay=0, select_stage_delay=0, level_start_delay=0,
            gameplay_start_delay=0,
        )
        self.assertEqual(backend.buttons, ["fullscreen", "start", "start", "start"])

    def test_native_frame_to_board(self):
        expected = np.array([[3, 4, 3], [1, 3, 1], [2, 1, 2]], dtype=np.int8)
        observation = BsnesNativeDetector().detect(cookie_frame(expected))
        np.testing.assert_array_equal(observation.board, expected)

    def test_rejects_wrong_resolution(self):
        with self.assertRaisesRegex(ValueError, "256x224"):
            BsnesNativeDetector().detect(np.zeros((100, 100, 3), dtype=np.uint8))

    def test_unknown_cookie_is_not_treated_as_yoshi(self):
        unknown = np.full((2, 2), 5, dtype=np.int8)
        observation = BsnesNativeDetector().detect(cookie_frame(unknown))
        np.testing.assert_array_equal(observation.board, np.zeros((2, 2), dtype=np.int8))
        self.assertEqual(observation.confidence, 0.0)

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
        backend = YdotoolInputBackend(BSNES_KEYS)
        controller = BsnesController(backend)
        backend.tap = lambda _button: None
        cursor = controller.execute(Move(0, 0, Direction.LEFT), (0, 0), (3, 4))
        self.assertEqual(cursor, (0, 3))


class PersistentUInputTests(unittest.TestCase):
    def _backend(self):
        device = Mock()
        ecodes = types.SimpleNamespace(EV_KEY=1)
        module = types.SimpleNamespace(UInput=Mock(), ecodes=ecodes)
        patcher = patch.dict("sys.modules", {"evdev": module})
        patcher.start()
        self.addCleanup(patcher.stop)
        backend = PersistentUInputBackend(
            BSNES_KEYS, registration_delay=0, key_delay=0, uinput=device
        )
        return backend, device

    def test_reuses_device_and_emits_a_direction_chord_in_order(self):
        backend, device = self._backend()
        cursor = backend.execute(Move(0, 0, Direction.RIGHT), (0, 0), (3, 4))
        self.assertEqual(cursor, (0, 1))
        self.assertEqual(
            [call.args for call in device.write.call_args_list],
            [(1, BSNES_KEYS["a"], 1), (1, BSNES_KEYS["right"], 1),
             (1, BSNES_KEYS["right"], 0), (1, BSNES_KEYS["a"], 0)],
        )
        self.assertEqual(device.syn.call_count, 4)

    def test_context_manager_closes_device(self):
        backend, device = self._backend()
        with backend:
            backend.tap("screenshot")
        device.close.assert_called_once_with()
        with self.assertRaisesRegex(Exception, "cerrado"):
            backend.tap("screenshot")


if __name__ == "__main__":
    unittest.main()
