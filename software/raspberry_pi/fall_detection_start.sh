#!/bin/bash
set -u

export DISPLAY=:0
export WAYLAND_DISPLAY=wayland-0
export XDG_RUNTIME_DIR=/run/user/1000
export DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus

systemctl --user start fall-detection.service

for _ in $(seq 1 100); do
    if systemctl --user is-active --quiet fall-detection.service; then
        /usr/bin/zenity --info \
            --title="Fall Detection" \
            --width=420 \
            --text="Fall detection started.

The camera window may take about 30 to 60 seconds to appear while the models load."
        exit 0
    fi
    sleep 0.1
done

/usr/bin/zenity --error \
    --title="Fall Detection" \
    --width=420 \
    --text="Failed to start fall detection."
exit 1
