"""HUD lateral opcional y estadísticas persistentes del kiosco."""

from __future__ import annotations

import json
import multiprocessing
import os
import queue
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from autoplay.domain import CookieType, Move


@dataclass
class HudSnapshot:
    state: str = "BOOTING"
    stage: int = 1
    moves: int = 0
    board: Optional[list[list[int]]] = None
    last_move: str = "--"
    tactical_score: float = 0.0
    error: Optional[str] = None


DEFAULT_STATS = {
    "sessions": 0,
    "total_moves": 0,
    "max_stage": 1,
    "max_tactical_score": 0,
    "longest_session_moves": 0,
    "errors": 0,
    "last_error": None,
    "updated_at": None,
}


class StatsStore:
    """JSON pequeño con reemplazo atómico para sobrevivir cierres abruptos."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.data = dict(DEFAULT_STATS)
        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
            self.data.update({key: loaded[key] for key in DEFAULT_STATS if key in loaded})
        except FileNotFoundError:
            pass
        except (OSError, ValueError, TypeError):
            # Un histórico roto no puede impedir que el bot juegue.
            self.data["last_error"] = "stats.json ilegible; histórico reiniciado"

    def begin_session(self) -> None:
        self.data["sessions"] += 1
        self.save()

    def record_move(self, session_moves: int, tactical_score: float) -> None:
        self.data["total_moves"] += 1
        self.data["longest_session_moves"] = max(
            self.data["longest_session_moves"], session_moves
        )
        self.data["max_tactical_score"] = max(
            self.data["max_tactical_score"], int(tactical_score)
        )
        self.save()

    def record_stage(self, stage: int) -> None:
        self.data["max_stage"] = max(self.data["max_stage"], stage)
        self.save()

    def record_error(self, message: str) -> None:
        self.data["errors"] += 1
        self.data["last_error"] = message
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data["updated_at"] = datetime.now(timezone.utc).isoformat()
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, self.path)


class HudDisplay:
    """Dos ventanas laterales; los fallos visuales no afectan al autoplay."""

    def __init__(self, title: str = "YOSHI AUTOPLAY MONITOR"):
        self.title = title
        self.snapshot = HudSnapshot()
        self.stats = dict(DEFAULT_STATS)
        self.started_at = time.monotonic()
        self.error: Optional[Exception] = None
        context = multiprocessing.get_context("spawn")
        self._updates = context.Queue()
        self._status = context.Queue()
        self._process = context.Process(
            target=self._run, args=(self.title, self._updates, self._status), daemon=True
        )

    def start(self) -> None:
        self._process.start()
        try:
            status = self._status.get(timeout=2.0)
            if status:
                self.error = RuntimeError(status)
        except queue.Empty:
            self.error = RuntimeError("el HUD no confirmó su inicio")

    def update(self, snapshot: HudSnapshot, stats: dict) -> None:
        if self._process.is_alive():
            self._updates.put((asdict(snapshot), dict(stats)))

    def close(self) -> None:
        if not self._process.is_alive():
            return
        self._updates.put(None)
        self._process.join(timeout=2.0)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=1.0)

    @staticmethod
    def _run(title, updates, status) -> None:
        try:
            import tkinter as tk

            renderer = object.__new__(HudDisplay)
            renderer.title = title
            renderer.started_at = time.monotonic()
            root = tk.Tk()
            root.withdraw()
            screen_w, screen_h = root.winfo_screenwidth(), root.winfo_screenheight()
            game_w = min(screen_w, round(screen_h * 4 / 3))
            side_w = max(180, (screen_w - game_w) // 2)
            left = renderer._panel(tk, root, 0, side_w, screen_h)
            right = renderer._panel(tk, root, screen_w - side_w, side_w, screen_h)
            left_label = renderer._label(tk, left, "#32ff72")
            right_label = renderer._label(tk, right, "#ff9d2e")
            snapshot, stats = HudSnapshot(), dict(DEFAULT_STATS)
            status.put(None)

            def refresh():
                nonlocal snapshot, stats
                try:
                    while True:
                        update = updates.get_nowait()
                        if update is None:
                            root.destroy()
                            return
                        snapshot_data, stats = update
                        snapshot = HudSnapshot(**snapshot_data)
                except queue.Empty:
                    pass
                left_label.configure(text=renderer._left_text(snapshot))
                right_label.configure(text=renderer._right_text(snapshot, stats))
                root.after(150, refresh)

            refresh()
            root.mainloop()
        except Exception as exc:  # pragma: no cover - depende del compositor
            status.put(str(exc))

    @staticmethod
    def _panel(tk, root, x: int, width: int, height: int):
        window = tk.Toplevel(root)
        window.overrideredirect(True)
        window.configure(bg="#020b06")
        window.geometry(f"{width}x{height}+{x}+0")
        window.attributes("-topmost", True)
        try:
            window.attributes("-type", "dock")
        except tk.TclError:
            pass
        return window

    @staticmethod
    def _label(tk, window, color: str):
        label = tk.Label(
            window, bg="#020b06", fg=color, anchor="nw", justify="left",
            padx=14, pady=20, font=("DejaVu Sans Mono", 10), takefocus=0,
        )
        label.pack(fill="both", expand=True)
        return label

    def _left_text(self, snapshot: HudSnapshot) -> str:
        uptime = int(time.monotonic() - self.started_at)
        board = self._board_text(snapshot.board)
        return (
            "YOSHI AUTOPLAY\n"
            "==============\n"
            "LIVE TELEMETRY\n\n"
            f"STATE  {snapshot.state}\n"
            f"STAGE  {snapshot.stage:03d}\n"
            f"MOVE   {snapshot.moves:06d}\n"
            f"UPTIME {uptime // 60:03d}:{uptime % 60:02d}\n\n"
            "BOARD ARRAY\n"
            "-----------\n"
            f"{board}\n\n"
            "LAST MOVE\n"
            f"{snapshot.last_move}\n"
            f"TACTICAL SCORE {snapshot.tactical_score:.0f}\n\n"
            "AUTOPLAY ACTIVE"
        )

    @staticmethod
    def _right_text(snapshot: HudSnapshot, stats: dict) -> str:
        error = snapshot.error or stats.get("last_error") or "NONE"
        return (
            "SESSION HISTORY\n"
            "===============\n"
            "PERSISTENT STATS\n\n"
            f"SESSIONS       {stats.get('sessions', 0):07d}\n"
            f"TOTAL MOVES    {stats.get('total_moves', 0):07d}\n"
            f"MAX STAGE      {stats.get('max_stage', 1):07d}\n"
            f"LONGEST RUN    {stats.get('longest_session_moves', 0):07d}\n"
            f"MAX TACTICAL   {stats.get('max_tactical_score', 0):07d}\n"
            f"ANOMALIES      {stats.get('errors', 0):07d}\n\n"
            "CURRENT SESSION\n"
            "---------------\n"
            f"STAGE           {snapshot.stage:07d}\n"
            f"MOVES           {snapshot.moves:07d}\n\n"
            "LAST ERROR\n"
            f"{error[:80]}\n\n"
            "MONITORING ENABLED"
        )

    @staticmethod
    def _board_text(board: Optional[list[list[int]]]) -> str:
        if not board:
            return "< ACQUIRING >"
        glyphs = {
            CookieType.UNKNOWN: "?", CookieType.DIAMOND: "D",
            CookieType.HEART: "H", CookieType.FLOWER: "F",
            CookieType.CHECKER: "X", CookieType.CIRCLE: "O",
            CookieType.YOSHI: "Y",
        }
        return "\n".join(" ".join(glyphs.get(value, "?") for value in row) for row in board)


class KioskTelemetry:
    def __init__(self, stats_path: Path, display: Optional[HudDisplay] = None):
        self.store = StatsStore(stats_path)
        self.display = display
        self.snapshot = HudSnapshot()

    def start(self) -> None:
        self.store.begin_session()
        if self.display:
            self.display.start()
        self.set_state("AWAITING SIGNAL")

    def restore_after_fullscreen(self) -> None:
        """Recrea el HUD por encima de una ventana que acaba de entrar en fullscreen."""
        if not self.display:
            return
        title = self.display.title
        self.display.close()
        self.display = HudDisplay(title)
        self.display.start()
        self._publish()

    def set_state(self, state: str, board: Optional[np.ndarray] = None) -> None:
        self.snapshot.state = state
        if board is not None:
            self.snapshot.board = board.astype(int).tolist()
        self._publish()

    def stage_start(self) -> None:
        self.snapshot.stage += 1
        self.snapshot.state = "STAGE TRANSITION"
        self.store.record_stage(self.snapshot.stage)
        self._publish()

    def move(self, number: int, board: np.ndarray, move: Move) -> None:
        self.snapshot.state = "MOVE VERIFIED"
        self.snapshot.moves = number
        self.snapshot.board = board.astype(int).tolist()
        self.snapshot.last_move = f"{move.direction.value.upper()} / AXIS {move.axis_index}"
        self.snapshot.tactical_score = move.score
        self.store.record_move(number, move.score)
        self._publish()

    def error(self, message: str) -> None:
        self.snapshot.state = "CONTAINMENT MODE"
        self.snapshot.error = message
        self.store.record_error(message)
        self._publish()

    def close(self) -> None:
        self.set_state("OFFLINE")
        if self.display:
            time.sleep(0.15)
            self.display.close()

    def _publish(self) -> None:
        if self.display:
            self.display.update(self.snapshot, self.store.data)
