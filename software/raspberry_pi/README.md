# Raspberry Pi software

The main application is `real_time_fall_detection.py`.

It combines:

- Picamera2 capture
- YOLO Pose skeleton extraction
- 100-frame TCN inference
- 20-frame GRU window generation
- asynchronous UART communication with Basys3
- local VNC status display
- combined TCN + FPGA emergency alert

Copy `.env.example` to `.env` and provide the two model paths before running.
See [SETUP_AND_RUN.md](../../docs/SETUP_AND_RUN.md).

