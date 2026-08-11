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
from autoplay.bsnes import (BSNES_KEYS, BsnesController, BsnesNativeDetector,
                            BsnesScreenStateError, BsnesScreenshotSource,
                            BsnesUnknownCookieError)
from autoplay.domain import Direction, Move
from autoplay.kiosk import KioskRunner
from autoplay.orchestrator import PostMoveVerificationError


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

    def test_detects_stage_start_screen(self):
        frame = np.zeros((224, 256, 3), dtype=np.uint8)
        for (x, y), color in (
            ((96, 96), (90, 255, 255)), ((96, 112), (16, 33, 24)),
            ((96, 128), (156, 156, 0)), ((128, 160), (0, 247, 8)),
        ):
            frame[y, x] = color
        with self.assertRaises(BsnesScreenStateError) as raised:
            BsnesNativeDetector().detect(frame)
        self.assertEqual(raised.exception.state, "stage_start")

    def test_unknown_cookie_is_not_treated_as_yoshi(self):
        unknown = np.full((2, 2), 5, dtype=np.int8)
        observation = BsnesNativeDetector().detect(cookie_frame(unknown))
        np.testing.assert_array_equal(observation.board, np.zeros((2, 2), dtype=np.int8))
        self.assertEqual(observation.confidence, 0.0)

    def test_detects_cursor_and_classifies_visible_symbol_around_it(self):
        expected = np.full((2, 2), 3, dtype=np.int8)
        frame = cookie_frame(expected)
        detector = BsnesNativeDetector()
        x, y = detector.LEFT + detector.CELL, detector.BOTTOM - detector.CELL
        frame[y - 3:y + 4, x - 3:x + 4] = (255, 255, 255)
        for dx, dy in ((-5, 0), (5, 0), (0, -5), (0, 5)):
            frame[y + dy, x + dx] = (0, 66, 255)
        observation = detector.detect(frame)
        self.assertEqual(observation.cursor, (0, 1))
        self.assertEqual(observation.board[0, 1], 3)

    def test_detects_dark_cursor_blink_phase_over_heart(self):
        expected = np.full((2, 2), 2, dtype=np.int8)
        frame = cookie_frame(expected)
        detector = BsnesNativeDetector()
        x, y = detector.LEFT + detector.CELL, detector.BOTTOM
        frame[y - 4:y + 5, x - 4:x + 5] = (20, 20, 20)
        lime_points = [(dx, dy) for dy in range(-3, 4) for dx in range(-3, 4)][:14]
        for dx, dy in lime_points:
            frame[y + dy, x + dx] = (90, 255, 255)
        for dx, dy in ((-5, 0), (5, 0), (0, -5), (0, 5)):
            frame[y + dy, x + dx] = (0, 0, 198)
        observation = detector.detect(frame)
        self.assertEqual(observation.cursor, (1, 1))
        self.assertEqual(observation.board[1, 1], 2)

    def test_checker_is_classified_in_both_sprite_orientations(self):
        expected = np.full((2, 2), 4, dtype=np.int8)
        frame = cookie_frame(expected)
        detector = BsnesNativeDetector()
        x, y = detector.LEFT, detector.BOTTOM - detector.CELL
        frame[y, x] = (165, 255, 255)
        points = [(dx, dy) for dy in range(-5, 6) for dx in range(-5, 6)][:20]
        for dx, dy in points:
            frame[y + dy, x + dx] = (0, 0, 255)
        observation = detector.detect(frame)
        np.testing.assert_array_equal(observation.board, expected)

    def test_yoshi_cookie_has_a_distinct_dark_and_lime_symbol(self):
        frame = cookie_frame(np.full((2, 2), 3, dtype=np.int8))
        detector = BsnesNativeDetector()
        x, y = detector.LEFT + detector.CELL, detector.BOTTOM
        frame[y - 5:y + 6, x - 5:x + 6] = (20, 20, 20)
        lime_points = [(dx, dy) for dy in range(-4, 5) for dx in range(-4, 5)][:22]
        for dx, dy in lime_points:
            frame[y + dy, x + dx] = (90, 255, 255)
        observation = detector.detect(frame)
        self.assertEqual(observation.board[1, 1], 6)
        self.assertEqual(observation.confidence, 1.0)

    def test_flower_is_classified_when_its_center_is_yellow(self):
        frame = cookie_frame(np.full((2, 2), 3, dtype=np.int8))
        detector = BsnesNativeDetector()
        x, y = detector.LEFT + detector.CELL, detector.BOTTOM
        frame[y, x] = (165, 255, 255)
        orange_points = [(dx, dy) for dy in range(-5, 6) for dx in range(-5, 6)][:15]
        for dx, dy in orange_points:
            frame[y + dy, x + dx] = (0, 66, 255)
        observation = detector.detect(frame)
        self.assertEqual(observation.board[1, 1], 3)

    def test_plain_brown_cookie_is_circle(self):
        frame = cookie_frame(np.full((2, 2), 3, dtype=np.int8))
        detector = BsnesNativeDetector()
        x, y = detector.LEFT + detector.CELL, detector.BOTTOM
        frame[y, x] = (16, 132, 206)
        observation = detector.detect(frame)
        self.assertEqual(observation.board[1, 1], 5)

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
            BSNES_KEYS, registration_delay=0, key_delay=0,
            navigation_hold=0, navigation_delay=0, uinput=device
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

    def test_cursor_navigation_holds_each_direction(self):
        backend, device = self._backend()
        cursor = backend.move_cursor((1, 2), (0, 0))
        self.assertEqual(cursor, (0, 0))
        self.assertEqual(
            [call.args[1:] for call in device.write.call_args_list],
            [(BSNES_KEYS["up"], 1), (BSNES_KEYS["up"], 0),
             (BSNES_KEYS["left"], 1), (BSNES_KEYS["left"], 0),
             (BSNES_KEYS["left"], 1), (BSNES_KEYS["left"], 0)],
        )


class KioskRunnerTests(unittest.TestCase):
    def test_unknown_cookie_stops_without_recovery(self):
        class Loop:
            def step(self, execute):
                raise BsnesUnknownCookieError([(1, 2)], Path("runtime/example.png"))

        controller = FakeInput()
        runner = KioskRunner(Loop(), controller, Path("/tmp/nonexistent-yca-stop"))
        self.assertEqual(runner.run(max_moves=5), 0)
        self.assertFalse(runner.running)
        self.assertEqual(controller.buttons, [])

    def test_post_move_verification_error_is_fatal(self):
        class Loop:
            def step(self, execute):
                raise PostMoveVerificationError("resultado ambiguo")

        controller = FakeInput()
        runner = KioskRunner(Loop(), controller, Path("/tmp/nonexistent-yca-stop"))
        self.assertEqual(runner.run(max_moves=5), 0)
        self.assertFalse(runner.running)
        self.assertEqual(controller.buttons, [])


if __name__ == "__main__":
    unittest.main()
