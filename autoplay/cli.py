"""CLI no interactiva y segura para Snes9x en KDE Wayland."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autoplay.adapters import WaylandFrameSource, YdotoolInputBackend
from autoplay.orchestrator import AutoPlayLoop, CycleConfig, Observation
from config import CONF
from main import ImprovedCookieDetector


DEFAULT_KEYS = {"up": 63, "down": 64, "left": 65, "right": 66, "a": 67}


def parse_region(value: str):
    try:
        result = tuple(int(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Usa x,y,ancho,alto") from exc
    if len(result) != 4 or result[2] <= 0 or result[3] <= 0:
        raise argparse.ArgumentTypeError("Usa x,y,ancho,alto con tamaño positivo")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Yoshi's Cookie autoplayer para Snes9x/Wayland")
    parser.add_argument("mode", choices=("observe", "single-step", "auto"))
    parser.add_argument("--region", required=True, type=parse_region, help="x,y,ancho,alto de Snes9x")
    parser.add_argument("--capture", choices=("auto", "grim", "spectacle"), default="auto")
    parser.add_argument("--interval", type=float, default=0.25)
    parser.add_argument("--stable-frames", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("--animation-delay", type=float, default=0.8)
    parser.add_argument("--max-moves", type=int, default=100)
    parser.add_argument("--keys", type=Path, help="JSON con keycodes evdev: up/down/left/right/a")
    parser.add_argument("--yes-really-execute", action="store_true",
                        help="Confirmación obligatoria para enviar teclas")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode != "observe" and not args.yes_really_execute:
        raise SystemExit("Ejecución bloqueada: agrega --yes-really-execute después de probar observe")

    frame_source = WaylandFrameSource(args.region, args.capture)
    detector = ImprovedCookieDetector(CONF)

    def observe():
        cookies = detector.detectar_cookies_image(frame_source.capture())
        board, info = detector.construir_grilla_inteligente(cookies)
        if not board.size:
            raise RuntimeError("No se detectó un tablero")
        return Observation(board, float(info.get("confianza", 0.0)))

    executor = None
    if args.mode != "observe":
        keys = DEFAULT_KEYS.copy()
        if args.keys:
            keys.update(json.loads(args.keys.read_text()))
        executor = YdotoolInputBackend(keys)
    config = CycleConfig(args.interval, args.stable_frames, args.animation_delay, args.timeout)
    loop = AutoPlayLoop(observe, executor=executor, config=config)

    limit = 1 if args.mode != "auto" else args.max_moves
    for number in range(1, limit + 1):
        before, move, after = loop.step(execute=args.mode != "observe")
        print(f"#{number} tablero={before.board.tolist()} movimiento={move.direction.value} "
              f"indice={move.axis_index} score={move.score:.0f}")
        if args.mode == "observe":
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
