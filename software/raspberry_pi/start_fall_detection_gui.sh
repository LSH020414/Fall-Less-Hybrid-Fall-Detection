#!/bin/bash
set -eu

export DISPLAY=:0
export WAYLAND_DISPLAY=wayland-0
export XDG_RUNTIME_DIR=/run/user/1000
export DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus

for _ in $(seq 1 120); do
    if [ -S /run/user/1000/wayland-0 ] || [ -S /tmp/.X11-unix/X0 ]; then
        break
    fi
    sleep 1
done

cd "${HOME}/fall_detection"
exec "${HOME}/fall_detection/yolovenv/bin/python" \
    "${HOME}/fall_detection/real_time_fall_detection.py"
