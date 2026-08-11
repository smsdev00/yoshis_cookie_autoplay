#!/usr/bin/env bash
set -euo pipefail

display_number="${DISPLAY#:}"
display_number="${display_number%%.*}"
lock_file="/tmp/.X${display_number}-lock"
rm -f -- "$lock_file"

Xvfb "$DISPLAY" \
    -screen 0 "${XVFB_SCREEN:-1280x960x24}" \
    -nolisten tcp \
    -noreset &

xvfb_pid=$!
for _ in $(seq 1 50); do
    if xdpyinfo -display "$DISPLAY" >/dev/null 2>&1; then
        exec "$@"
    fi
    if ! kill -0 "$xvfb_pid" 2>/dev/null; then
        echo "Xvfb terminó antes de aceptar conexiones en $DISPLAY" >&2
        exit 1
    fi
    sleep 0.1
done

echo "Xvfb no quedó disponible en $DISPLAY" >&2
exit 1

