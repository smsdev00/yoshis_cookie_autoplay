"""Modo autónomo para ejecutar Yoshi's Cookie con bsnes."""

from __future__ import annotations

import argparse
import signal
import subprocess
import time
from pathlib import Path

import cv2

from autoplay.adapters import PersistentUInputBackend
from autoplay.bsnes import (
    BSNES_KEYS,
    DEFAULT_BSNES,
    DEFAULT_ROM,
    DEFAULT_SCREENSHOT_DIR,
    DEFAULT_SCREENSHOT_GLOB,
    BsnesController,
    BsnesNativeDetector,
    BsnesProcess,
    BsnesScreenStateError,
    BsnesScreenshotSource,
)
from autoplay.orchestrator import AutoPlayLoop, CycleConfig
from autoplay.solver import Solver


class KioskRunner:
    def __init__(self, loop: AutoPlayLoop, controller: BsnesController,
                 stop_file: Path, recovery_failures: int = 5,
                 recovery_starts: int = 2):
        self.loop = loop
        self.controller = controller
        self.stop_file = stop_file
        self.recovery_failures = recovery_failures
        self.recovery_starts = recovery_starts
        self.running = True

    def stop(self, *_args) -> None:
        self.running = False

    def run(self, max_moves: int = 0) -> int:
        failures = moves = 0
        while self.running and not self.stop_file.exists():
            if max_moves and moves >= max_moves:
                break
            try:
                before, move, _after = self.loop.step(execute=True)
                moves += 1
                failures = 0
                print(f"#{moves} {before.board.shape} {move.direction.value} "
                      f"índice={move.axis_index} score={move.score:.0f}", flush=True)
            except (TimeoutError, RuntimeError, ValueError) as exc:
                failures += 1
                print(f"[WARN] observación #{failures} falló: {exc}", flush=True)
                if failures >= self.recovery_failures and self.running:
                    self._recover()
                    failures = 0
                else:
                    time.sleep(0.5)
        return moves

    def _recover(self) -> None:
        print("[INFO] intentando salir de Game Over/título con Start", flush=True)
        for _ in range(self.recovery_starts):
            self.controller.tap("start")
            time.sleep(1.0)
        self.loop.cursor = (0, 0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Yoshi's Cookie como kiosco con bsnes")
    parser.add_argument(
        "mode", choices=("inspect", "observe", "cursor-check", "single-step", "kiosk")
    )
    parser.add_argument("--bsnes", type=Path, default=DEFAULT_BSNES)
    parser.add_argument("--rom", type=Path, default=DEFAULT_ROM)
    parser.add_argument("--screenshots", type=Path, default=DEFAULT_SCREENSHOT_DIR)
    parser.add_argument("--pattern", default=DEFAULT_SCREENSHOT_GLOB)
    parser.add_argument("--image", type=Path, help="BMP existente para inspect")
    parser.add_argument("--launch", action="store_true", help="Iniciar bsnes, ROM y fullscreen")
    parser.add_argument("--launch-delay", type=float, default=20.0,
                        help="Espera antes de fullscreen/Start")
    parser.add_argument("--select-stage-delay", type=float, default=8.0,
                        help="Espera desde fullscreen hasta Select Stage")
    parser.add_argument("--level-start-delay", type=float, default=10.0,
                        help="Espera desde Select Stage hasta Start Level")
    parser.add_argument("--gameplay-start-delay", type=float, default=10.0,
                        help="Espera hasta confirmar PUSH START")
    parser.add_argument("--max-moves", type=int, default=0, help="0 significa sin límite")
    parser.add_argument("--stop-file", type=Path, default=Path("runtime/STOP"))
    parser.add_argument("--interval", type=float, default=0.25)
    parser.add_argument("--stable-frames", type=int, default=2)
    parser.add_argument("--animation-delay", type=float, default=0.8)
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("--yes-really-execute", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode not in ("inspect", "observe") and not args.yes_really_execute:
        raise SystemExit("Agrega --yes-really-execute después de validar el modo observe")

    detector = BsnesNativeDetector()
    if args.mode == "inspect":
        candidates = list(args.screenshots.expanduser().glob(args.pattern))
        if not args.image and not candidates:
            raise SystemExit(f"No hay capturas que coincidan con {args.pattern}")
        path = args.image.expanduser() if args.image else max(
            candidates, key=lambda item: item.stat().st_mtime_ns
        )
        observation = detector.detect(cv2.imread(str(path), cv2.IMREAD_COLOR))
        move = Solver().best_move(observation.board)
        print(f"archivo={path}")
        print(f"tablero={observation.board.tolist()}")
        print(f"propuesta={move.direction.value} índice={move.axis_index} score={move.score:.0f}")
        return 0

    input_backend = PersistentUInputBackend(BSNES_KEYS)
    controller = BsnesController(input_backend)
    process = BsnesProcess(args.bsnes, args.rom)
    try:
        if args.launch:
            process.launch()
            controller.prepare(
                launch_delay=args.launch_delay,
                select_stage_delay=args.select_stage_delay,
                level_start_delay=args.level_start_delay,
                gameplay_start_delay=args.gameplay_start_delay,
            )

        source = BsnesScreenshotSource(controller, args.screenshots, args.pattern)
        config = CycleConfig(
            poll_interval=args.interval,
            stable_frames=args.stable_frames,
            animation_delay=args.animation_delay,
            stability_timeout=args.timeout,
            min_confidence=0.98,
            require_cursor=True,
        )
        last_stage_start = 0.0

        def observe_board():
            nonlocal last_stage_start
            try:
                return detector.detect(source.capture())
            except BsnesScreenStateError as exc:
                now = time.monotonic()
                if (args.mode == "kiosk" and exc.state == "stage_start" and
                        now - last_stage_start >= 5.0):
                    print("[INFO] Stage Start detectado; enviando Keypad8", flush=True)
                    controller.tap("start")
                    last_stage_start = now
                    time.sleep(3.0)
                raise

        loop = AutoPlayLoop(observe_board, executor=controller, config=config)

        if args.mode == "observe":
            observation, move = loop.propose()
            print(f"tablero={observation.board.tolist()}")
            print(f"propuesta={move.direction.value} índice={move.axis_index} score={move.score:.0f}")
            return 0
        if args.mode == "cursor-check":
            before = loop.wait_for_stable_board()
            target = (0, 0)
            loop.cursor = before.cursor
            loop.position_cursor(target, before.board.shape)
            print(f"cursor_antes={before.cursor}")
            print(f"cursor_después={loop.cursor}")
            return 0
        if args.mode == "single-step":
            before, move, after = loop.step(execute=True)
            print(f"antes={before.board.tolist()}")
            print(f"movimiento={move.direction.value} índice={move.axis_index}")
            print(f"después={after.board.tolist() if after else None}")
            return 0

        args.stop_file.parent.mkdir(parents=True, exist_ok=True)
        runner = KioskRunner(loop, controller, args.stop_file)
        signal.signal(signal.SIGINT, runner.stop)
        signal.signal(signal.SIGTERM, runner.stop)
        print(f"[INFO] kiosco activo; detén con Ctrl+C o creando {args.stop_file}", flush=True)
        runner.run(args.max_moves)
    finally:
        if args.launch and process.running():
            process.process.terminate()
            try:
                process.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.process.kill()
        input_backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
