"""Adaptadores de captura Wayland y entrada virtual, aislados del solver."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Protocol, Tuple

import cv2
import numpy as np

from autoplay.domain import Direction, Move

Region = Tuple[int, int, int, int]


class CaptureError(RuntimeError):
    pass


class InputError(RuntimeError):
    pass


class InputBackend(Protocol):
    """Contrato mínimo para controlar bsnes sin acoplarlo a una herramienta."""

    def tap(self, button: str) -> None:
        ...

    def execute(self, move: Move, cursor: Tuple[int, int],
                board_shape: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
        ...


class PersistentUInputBackend:
    """Teclado evdev persistente para que Wayland registre el dispositivo."""

    def __init__(self, keycodes: Mapping[str, int], registration_delay: float = 1.0,
                 key_delay: float = 0.03, device_name: str = "Yoshi Cookie AutoPlayer",
                 uinput=None):
        self.keycodes = dict(keycodes)
        self.key_delay = key_delay
        self._closed = False
        try:
            from evdev import UInput, ecodes
        except ImportError as exc:
            raise InputError(
                "Falta python-evdev; instala las dependencias de requirements.txt"
            ) from exc

        self._ecodes = ecodes
        try:
            self._device = uinput or UInput(
                {ecodes.EV_KEY: sorted(set(self.keycodes.values()))},
                name=device_name,
                version=0x3,
            )
        except (OSError, PermissionError) as exc:
            raise InputError(
                f"No se pudo abrir /dev/uinput: {exc}. "
                "Carga uinput y concede acceso limitado al usuario."
            ) from exc
        if registration_delay > 0:
            time.sleep(registration_delay)

    def _emit(self, code: int, value: int) -> None:
        if self._closed:
            raise InputError("El dispositivo uinput ya está cerrado")
        try:
            self._device.write(self._ecodes.EV_KEY, code, value)
            self._device.syn()
        except OSError as exc:
            raise InputError(f"Falló el envío por uinput: {exc}") from exc
        if self.key_delay > 0:
            time.sleep(self.key_delay)

    def tap(self, button: str) -> None:
        try:
            code = self.keycodes[button]
        except KeyError as exc:
            raise InputError(f"Botón sin mapear: {button}") from exc
        self._emit(code, 1)
        self._emit(code, 0)

    def execute(self, move: Move, cursor: Tuple[int, int],
                board_shape: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
        row, col = _move_cursor_to(self, cursor, (move.row, move.col))
        try:
            a = self.keycodes["a"]
            direction = self.keycodes[move.direction.value]
        except KeyError as exc:
            raise InputError(f"Botón sin mapear: {exc.args[0]}") from exc
        self._emit(a, 1)
        try:
            self._emit(direction, 1)
            self._emit(direction, 0)
        finally:
            self._emit(a, 0)
        return _shifted_cursor((row, col), move, board_shape)

    def close(self) -> None:
        if not self._closed:
            self._device.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()


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


def _move_cursor_to(backend, cursor: Tuple[int, int],
                    target: Tuple[int, int]) -> Tuple[int, int]:
    row, col = cursor
    target_row, target_col = target
    while row < target_row:
        backend.tap("down"); row += 1
    while row > target_row:
        backend.tap("up"); row -= 1
    while col < target_col:
        backend.tap("right"); col += 1
    while col > target_col:
        backend.tap("left"); col -= 1
    return row, col


def _shifted_cursor(cursor: Tuple[int, int], move: Move,
                    board_shape: Optional[Tuple[int, int]]) -> Tuple[int, int]:
    row, col = cursor
    if board_shape:
        rows, cols = board_shape
        if move.direction == Direction.LEFT:
            col = (col - 1) % cols
        elif move.direction == Direction.RIGHT:
            col = (col + 1) % cols
        elif move.direction == Direction.UP:
            row = (row - 1) % rows
        elif move.direction == Direction.DOWN:
            row = (row + 1) % rows
    return row, col


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

    def execute(self, move: Move, cursor: Tuple[int, int],
                board_shape: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
        row, col = _move_cursor_to(self, cursor, (move.row, move.col))

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
        # El juego desplaza también la cookie seleccionada y el cursor.
        return _shifted_cursor((row, col), move, board_shape)
