"""Integración con bsnes: framebuffer BMP, proceso y controles de teclado."""

from __future__ import annotations

import subprocess
import time
from math import ceil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from autoplay.adapters import InputBackend
from autoplay.domain import CookieType
from autoplay.orchestrator import Observation


DEFAULT_BSNES = Path("/home/sms/Documents/bsnes-nightly/bsnes")
DEFAULT_ROM = Path("/home/sms/Downloads/Yoshi's Cookie (USA).zip")
DEFAULT_SCREENSHOT_DIR = Path("/home/sms/Downloads")
DEFAULT_SCREENSHOT_GLOB = "Yoshi's Cookie (USA)-*.bmp"

# Códigos evdev de las teclas configuradas por el usuario en bsnes.
BSNES_KEYS = {
    "up": 17,         # W
    "down": 31,       # S
    "left": 30,       # A
    "right": 32,      # D
    "a": 24,          # O (botón A del mando SNES)
    "start": 72,      # Keypad8
    "fullscreen": 87, # F11
    "screenshot": 88, # F12
}


class BsnesScreenshotError(RuntimeError):
    pass


class BsnesScreenStateError(ValueError):
    def __init__(self, state: str):
        self.state = state
        super().__init__(f"Pantalla de bsnes detectada: {state}")


class BsnesUnknownCookieError(Exception):
    def __init__(self, positions, diagnostic: Path):
        self.positions = tuple(positions)
        self.diagnostic = diagnostic
        super().__init__(
            f"Cookies desconocidas en {self.positions}; diagnóstico={diagnostic}"
        )


class BsnesScreenshotSource:
    """Pide un screenshot con F12 y espera el BMP nuevo y completo."""

    def __init__(self, input_backend: InputBackend,
                 directory: Path = DEFAULT_SCREENSHOT_DIR,
                 pattern: str = DEFAULT_SCREENSHOT_GLOB,
                 timeout: float = 2.0, poll_interval: float = 0.02):
        self.input = input_backend
        self.directory = Path(directory).expanduser()
        self.pattern = pattern
        self.timeout = timeout
        self.poll_interval = poll_interval

    def capture(self) -> np.ndarray:
        before = {path: (path.stat().st_mtime_ns, path.stat().st_size)
                  for path in self.directory.glob(self.pattern)}
        self.input.tap("screenshot")
        path = self._wait_for_new_file(before)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise BsnesScreenshotError(f"bsnes produjo un BMP inválido: {path}")
        return image

    def _wait_for_new_file(self, before) -> Path:
        deadline = time.monotonic() + self.timeout
        candidate: Optional[Path] = None
        stable_size: Optional[int] = None
        stable_checks = 0
        while time.monotonic() < deadline:
            changed = []
            for path in self.directory.glob(self.pattern):
                try:
                    stat = path.stat()
                except FileNotFoundError:
                    continue
                old = before.get(path)
                if old is None or (stat.st_mtime_ns, stat.st_size) != old:
                    changed.append((stat.st_mtime_ns, path, stat.st_size))
            if changed:
                _, newest, size = max(changed)
                if newest == candidate and size == stable_size and size > 54:
                    stable_checks += 1
                else:
                    candidate, stable_size, stable_checks = newest, size, 1
                if stable_checks >= 2:
                    return newest
            time.sleep(self.poll_interval)
        raise BsnesScreenshotError(
            f"No apareció un BMP nuevo en {self.directory} tras pulsar F12"
        )


class BsnesNativeDetector:
    """Lee directamente el framebuffer 256×224 de Yoshi's Cookie (USA)."""

    WIDTH = 256
    HEIGHT = 224
    LEFT = 80
    BOTTOM = 184
    CELL = 16
    MAX_CELLS = 8
    MIN_COOKIE_PIXELS = 60

    def detect(self, image: np.ndarray) -> Observation:
        if image is None or image.shape[:2] != (self.HEIGHT, self.WIDTH):
            shape = None if image is None else image.shape[:2]
            raise ValueError(f"Se esperaba framebuffer 256x224; recibido {shape}")
        if self._is_stage_start(image):
            raise BsnesScreenStateError("stage_start")

        width = self._extent(image, horizontal=True)
        height = self._extent(image, horizontal=False)
        if width == 0 or height == 0 or width * height < 2:
            raise ValueError(f"Tablero no jugable o en animación: {height}x{width}")

        board = np.zeros((height, width), dtype=np.int8)
        scores = []
        cursor = None
        top = self.BOTTOM - (height - 1) * self.CELL
        for row in range(height):
            for col in range(width):
                x, y = self.LEFT + col * self.CELL, top + row * self.CELL
                score = self._cookie_score(image, x, y)
                scores.append(score)
                if score < self.MIN_COOKIE_PIXELS:
                    raise ValueError(f"Hueco inesperado en tablero rectangular ({row}, {col})")
                if self._has_cursor(image, x, y):
                    if cursor is not None:
                        raise ValueError("Se detectó más de un cursor en el tablero")
                    cursor = (row, col)
                    board[row, col] = self._classify_occluded(image, x, y)
                else:
                    board[row, col] = self._classify(image, x, y)

        confidence = 0.0 if np.any(board == CookieType.UNKNOWN) else min(
            1.0, min(scores) / self.MIN_COOKIE_PIXELS
        )
        return Observation(board, confidence, cursor)

    @staticmethod
    def _is_stage_start(image: np.ndarray) -> bool:
        """Reconoce el panel fijo STAGE START / PUSH START del framebuffer."""
        samples = (
            ((96, 96), (90, 255, 255)),
            ((96, 112), (16, 33, 24)),
            ((96, 128), (156, 156, 0)),
            ((128, 160), (0, 247, 8)),
        )
        return all(
            np.max(np.abs(image[y, x].astype(np.int16) - np.asarray(color))) <= 4
            for (x, y), color in samples
        )

    def _extent(self, image: np.ndarray, horizontal: bool) -> int:
        total = 0
        for index in range(self.MAX_CELLS):
            x = self.LEFT + index * self.CELL if horizontal else self.LEFT
            y = self.BOTTOM if horizontal else self.BOTTOM - index * self.CELL
            if self._cookie_score(image, x, y) < self.MIN_COOKIE_PIXELS:
                break
            total += 1
        return total

    @staticmethod
    def _cookie_score(image: np.ndarray, x: int, y: int) -> int:
        patch = image[y - 7:y + 8, x - 7:x + 8]
        # La masa dorada común a las cinco cookies del framebuffer SNES.
        mask = ((patch[:, :, 2] > 180) & (patch[:, :, 1] > 120) &
                (patch[:, :, 0] < 220))
        return int(np.count_nonzero(mask))

    @staticmethod
    def _classify(image: np.ndarray, x: int, y: int) -> int:
        patch = image[y - 7:y + 8, x - 7:x + 8]
        blue_patch, green_patch, red_patch = cv2.split(patch)
        dark_red = int(np.count_nonzero((red_patch > 140) & (green_patch < 30)))
        orange = int(np.count_nonzero(
            (red_patch > 220) & (green_patch >= 30) & (green_patch < 130)
        ))
        core = patch[2:13, 2:13]
        core_black = int(np.count_nonzero(np.all(core < 40, axis=2)))
        lime = int(np.count_nonzero(
            (green_patch > 220) & (red_patch > 180) & (blue_patch < 130)
        ))
        if core_black >= 20 and lime >= 15:
            return CookieType.YOSHI
        # El símbolo ajedrezado alterna su orientación según la celda, por lo
        # que su píxel central no es estable. Ambas fases conservan esta firma.
        if 8 <= dark_red <= 35 and orange < 15:
            return CookieType.CHECKER
        if dark_red >= 40:
            return CookieType.HEART
        if orange >= 15:
            return CookieType.FLOWER
        if int(np.count_nonzero((green_patch > 220) & (red_patch < 80))) >= 15:
            return CookieType.DIAMOND
        blue, green, red = (int(value) for value in image[y, x])
        if green > 220 and red < 80:
            return CookieType.DIAMOND
        if red > 140 and green < 30:
            return CookieType.HEART
        if red > 220 and 30 <= green < 130:
            return CookieType.FLOWER
        if red > 180 and green > 180 and blue < 80:
            return CookieType.CHECKER
        if 180 <= red <= 230 and 90 <= green <= 170 and blue < 60:
            return CookieType.CIRCLE
        # Nunca adivinar Yoshi: un tipo desconocido debe impedir la ejecución.
        return CookieType.UNKNOWN

    @staticmethod
    def _has_cursor(image: np.ndarray, x: int, y: int) -> bool:
        core = image[y - 3:y + 4, x - 3:x + 4]
        white = np.all(core > 240, axis=2)
        if int(np.count_nonzero(white)) >= 4:
            return True
        patch = image[y - 7:y + 8, x - 7:x + 8]
        blue, green, red = cv2.split(patch)
        core_black = int(np.count_nonzero(np.all(patch[2:13, 2:13] < 40, axis=2)))
        lime = int(np.count_nonzero((green > 220) & (red > 180) & (blue < 130)))
        return core_black >= 20 and 8 <= lime < 18

    @staticmethod
    def _classify_occluded(image: np.ndarray, x: int, y: int) -> int:
        """Clasifica el símbolo que sobrevive alrededor de la mira del cursor."""
        patch = image[y - 7:y + 8, x - 7:x + 8]
        blue, green, red = cv2.split(patch)
        core_black = int(np.count_nonzero(np.all(patch[2:13, 2:13] < 40, axis=2)))
        lime = int(np.count_nonzero((green > 220) & (red > 180) & (blue < 130)))
        if core_black >= 10 and lime >= 20:
            return CookieType.YOSHI
        counts = {
            CookieType.DIAMOND: np.count_nonzero((green > 220) & (red < 80)),
            CookieType.HEART: np.count_nonzero((red > 140) & (green < 30)),
            CookieType.FLOWER: np.count_nonzero(
                (red > 220) & (green >= 30) & (green < 130)
            ),
        }
        cookie_type, pixels = max(counts.items(), key=lambda item: item[1])
        return cookie_type if pixels >= 2 else CookieType.UNKNOWN


@dataclass
class BsnesProcess:
    executable: Path = DEFAULT_BSNES
    rom: Path = DEFAULT_ROM
    process: Optional[subprocess.Popen] = None

    def launch(self) -> subprocess.Popen:
        executable = self.executable.expanduser()
        rom = self.rom.expanduser()
        if not executable.is_file():
            raise FileNotFoundError(f"No se encontró bsnes: {executable}")
        if not rom.is_file():
            raise FileNotFoundError(f"No se encontró la ROM: {rom}")
        self.process = subprocess.Popen(
            [str(executable), str(rom)],
            cwd=str(executable.parent),
            start_new_session=True,
        )
        return self.process

    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None


class BsnesController:
    """Controles de bsnes delegados a un backend de entrada intercambiable."""

    def __init__(self, input_backend: InputBackend):
        self.input = input_backend

    def tap(self, button: str) -> None:
        self.input.tap(button)

    def execute(self, move, cursor, board_shape=None):
        return self.input.execute(move, cursor, board_shape)

    def move_cursor(self, cursor, target):
        return self.input.move_cursor(cursor, target)

    def step_cursor(self, direction):
        self.input.step_cursor(direction)

    def prepare(self, launch_delay: float = 20.0, select_stage_delay: float = 8.0,
                level_start_delay: float = 10.0,
                gameplay_start_delay: float = 10.0) -> None:
        deadline = time.monotonic() + max(0.0, launch_delay)
        while True:
            remaining = max(0, ceil(deadline - time.monotonic()))
            print(f"\r[INFO] esperando que la ROM acepte entrada: {remaining:2d}s",
                  end="", flush=True)
            if remaining == 0:
                break
            time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
        print(flush=True)
        self.tap("fullscreen")
        print(f"[INFO] esperando {select_stage_delay:g}s hasta Select Stage", flush=True)
        time.sleep(max(0.0, select_stage_delay))
        print("[INFO] Select Stage: enviando Keypad8", flush=True)
        self.tap("start")
        print(f"[INFO] esperando {level_start_delay:g}s hasta Start Level", flush=True)
        time.sleep(max(0.0, level_start_delay))
        print("[INFO] Start Level: enviando Keypad8", flush=True)
        self.tap("start")
        print(f"[INFO] esperando {gameplay_start_delay:g}s hasta PUSH START", flush=True)
        time.sleep(max(0.0, gameplay_start_delay))
        print("[INFO] PUSH START: enviando Keypad8", flush=True)
        self.tap("start")
