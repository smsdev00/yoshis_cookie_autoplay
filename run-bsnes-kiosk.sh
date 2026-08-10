#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "$0")"

if [[ ! -x venv/bin/python ]]; then
    echo "Falta venv/bin/python; crea el entorno e instala requirements.txt" >&2
    exit 1
fi

if command -v systemd-inhibit >/dev/null 2>&1; then
    exec systemd-inhibit \
        --what=idle:sleep \
        --who="Yoshi's Cookie AutoPlayer" \
        --why="bsnes kiosk activo" \
        venv/bin/python -m autoplay.kiosk kiosk --launch --yes-really-execute "$@"
fi

exec venv/bin/python -m autoplay.kiosk kiosk --launch --yes-really-execute "$@"
