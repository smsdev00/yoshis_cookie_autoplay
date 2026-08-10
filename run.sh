#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "$0")"
source venv/bin/activate

if [[ "${1:-}" == "--gui" ]]; then
    shift
    export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"
    exec python3 main.py "$@"
fi

exec python3 main.py --headless "$@"
