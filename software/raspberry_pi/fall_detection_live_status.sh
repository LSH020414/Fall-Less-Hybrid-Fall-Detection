#!/bin/bash

export XDG_RUNTIME_DIR="/run/user/$(id -u)"
export DBUS_SESSION_BUS_ADDRESS="unix:path=${XDG_RUNTIME_DIR}/bus"

clear
echo "============================================================"
echo " Fall-Less Live Decision Pipeline"
echo "============================================================"
echo " FPGA GRU FALL and Pi5 TCN FALL are evaluated independently."
echo " CALL POLICE is produced only when both values are True."
echo " Press Ctrl+C to close this monitor."
echo "============================================================"
echo

exec journalctl --user -u fall-detection.service -f -o cat
