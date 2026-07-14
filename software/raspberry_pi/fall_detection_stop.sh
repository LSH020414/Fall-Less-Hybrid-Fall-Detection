#!/bin/bash
set -u

export DISPLAY=:0
export WAYLAND_DISPLAY=wayland-0
export XDG_RUNTIME_DIR=/run/user/1000
export DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus

systemctl --user stop fall-detection.service

if systemctl --user is-active --quiet fall-detection.service; then
    /usr/bin/zenity --error \
        --title="Fall Detection" \
        --width=420 \
        --text="Failed to stop fall detection."
    exit 1
fi

/usr/bin/zenity --info \
    --title="Fall Detection" \
    --width=420 \
    --text="Fall detection stopped.

Camera, YOLO/TCN, and FPGA UART transmission are now inactive."
