#!/bin/bash
set -u

APP_DIR="${HOME}/fall_detection"
FLAG="$APP_DIR/reset_alert.flag"

export DISPLAY=:0
export WAYLAND_DISPLAY=wayland-0
export XDG_RUNTIME_DIR=/run/user/1000
export DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus

touch "$FLAG"

for _ in $(seq 1 30); do
    if [ ! -e "$FLAG" ]; then
        break
    fi
    sleep 0.1
done

if systemctl --user is-active --quiet fall-detection.service; then
    status_text="Python/TCN reset request completed."
else
    status_text="Reset request saved. It will apply when detection starts."
fi

/usr/bin/zenity --info \
    --title="Fall Detection Reset" \
    --width=460 \
    --text="$status_text

Press BTNC on the Basys3 board once to clear the FPGA FALL latch."
