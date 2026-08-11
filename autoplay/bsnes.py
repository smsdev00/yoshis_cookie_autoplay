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

        width = self._extent(image, horizontal=True)
        height = self._extent(image, horizontal=False)
        if width < 2 or height < 2:
            raise ValueError(f"Tablero no jugable o en animación: {height}x{width}")

        board = np.zeros((height, width), dtype=np.int8)
        scores = []
        top = self.BOTTOM - (height - 1) * self.CELL
        for row in range(height):
            for col in range(width):
                x, y = self.LEFT + col * self.CELL, top + row * self.CELL
                score = self._cookie_score(image, x, y)
                scores.append(score)
                if score < self.MIN_COOKIE_PIXELS:
                    raise ValueError(f"Hueco inesperado en tablero rectangular ({row}, {col})")
                board[row, col] = self._classify(image, x, y)

        confidence = min(1.0, min(scores) / self.MIN_COOKIE_PIXELS)
        return Observation(board, confidence)

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
        blue, green, red = (int(value) for value in image[y, x])
        if green > 220 and red < 80:
            return 1  # diamante verde
        if red > 140 and green < 30:
            return 2  # corazón
        if red > 220 and 30 <= green < 130:
            return 3  # flor/círculo naranja
        if red > 180 and green > 180 and blue < 80:
            return 4  # check amarillo
        return 5      # Yoshi/comodín o tipo aún no visto


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

    def prepare(self, launch_delay: float = 20.0, startup_starts: int = 2) -> None:
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
        time.sleep(0.5)
        for _ in range(startup_starts):
            self.tap("start")
            time.sleep(1.0)
