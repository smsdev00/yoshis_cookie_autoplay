"""Adaptadores de captura Wayland y entrada virtual, aislados del solver."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Tuple

import cv2
import numpy as np

from autoplay.domain import Direction, Move

Region = Tuple[int, int, int, int]


class CaptureError(RuntimeError):
    pass


class InputError(RuntimeError):
    pass


class WaylandFrameSource:
    """Captura con grim y usa Spectacle como alternativa para KDE Plasma."""

    def __init__(self, region: Region, backend: str = "auto"):
        self.region = region
        self.requested_backend = backend
        self.backend = self._select_backend(backend)

    @staticmethod
    def _select_backend(requested: str) -> str:
        if requested != "auto":
            if not shutil.which(requested):
                raise CaptureError(f"No se encontró el backend de captura: {requested}")
            return requested
        for name in ("grim", "spectacle"):
            if shutil.which(name):
                return name
        raise CaptureError("Instala grim o spectacle para capturar bajo Wayland")

    def capture(self) -> np.ndarray:
        if self.backend == "grim":
            try:
                return self._grim()
            except CaptureError:
                if self.requested_backend != "auto" or not shutil.which("spectacle"):
                    raise
                self.backend = "spectacle"
        return self._spectacle()

    def _grim(self) -> np.ndarray:
        x, y, width, height = self.region
        proc = subprocess.run(
            ["grim", "-g", f"{x},{y} {width}x{height}", "-"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if proc.returncode:
            raise CaptureError(proc.stderr.decode(errors="replace").strip() or "grim falló")
        image = cv2.imdecode(np.frombuffer(proc.stdout, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise CaptureError("grim devolvió una imagen inválida")
        return image

    def _spectacle(self) -> np.ndarray:
        x, y, width, height = self.region
        with tempfile.TemporaryDirectory(prefix="yca-") as directory:
            path = Path(directory) / "screen.png"
            proc = subprocess.run(
                ["spectacle", "--background", "--nonotify", "--fullscreen", "--output", str(path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=False,
            )
            if proc.returncode or not path.exists():
                raise CaptureError(proc.stderr.decode(errors="replace").strip() or "Spectacle falló")
            full = cv2.imread(str(path))
        if full is None or x < 0 or y < 0 or x + width > full.shape[1] or y + height > full.shape[0]:
            raise CaptureError("La región configurada queda fuera de la captura")
        return full[y:y + height, x:x + width].copy()


@dataclass
class YdotoolInputBackend:
    """Envía botones SNES mapeados a códigos evdev mediante ydotool."""

    keycodes: Mapping[str, int]
    def __post_init__(self):
        if not shutil.which("ydotool"):
            raise InputError("No se encontró ydotool; el modo de ejecución no está disponible")

    def tap(self, button: str) -> None:
        code = self.keycodes[button]
        proc = subprocess.run(
            ["ydotool", "key", f"{code}:1", f"{code}:0"],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode:
            raise InputError(proc.stderr.strip() or f"ydotool falló enviando {button}")

    def execute(self, move: Move, cursor: Tuple[int, int]) -> Tuple[int, int]:
        row, col = cursor
        while row < move.row:
            self.tap("down"); row += 1
        while row > move.row:
            self.tap("up"); row -= 1
        while col < move.col:
            self.tap("right"); col += 1
        while col > move.col:
            self.tap("left"); col -= 1

        # El manual: mantener A y pulsar dirección. Un solo comando conserva
        # el orden press(A), tap(dirección), release(A).
        a = self.keycodes["a"]
        direction = self.keycodes[move.direction.value]
        proc = subprocess.run(
            ["ydotool", "key", f"{a}:1", f"{direction}:1", f"{direction}:0", f"{a}:0"],
            capture_output=True, text=True, check=False,
        )
        if proc.returncode:
            raise InputError(proc.stderr.strip() or "ydotool falló ejecutando el movimiento")
        return row, col
