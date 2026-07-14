# 설치 및 실행

## 1. Raspberry Pi 준비

권장 환경:

- Raspberry Pi 5 8GB
- Raspberry Pi OS 64-bit
- Python virtual environment
- Picamera2
- PyTorch
- Ultralytics YOLO
- OpenCV
- pyserial

```bash
cd ~/fall_detection
python3 -m venv --system-site-packages yolovenv
source yolovenv/bin/activate
pip install -r requirements.txt
```

`models/`에 다음 파일을 배치한다.

```text
yolo11n_blur_pose_best.pt
best_tcn_videowise.pt
```

## 2. 환경설정

```bash
cp .env.example .env
set -a
source .env
set +a
```

인터넷이 없는 Ethernet 직결 환경에서는 Telegram을 비활성화한다.

## 3. UART 활성화

`/boot/firmware/config.txt`:

```text
enable_uart=1
```

serial console을 비활성화한 뒤 `/dev/serial0`를 사용한다.

## 4. FPGA program

Vivado Hardware Manager에서 다음 bitstream을 지정한다.

```text
hardware/fpga_gru/bitstream/GRU_Fall_Detect_MODERATE_FINAL.bit
```

`GRU_Fall_Detect_TUNED115_FINAL.bit`은 118개 평가 record에 직접 최적화한
민감형 비교 프로필이다. 실시간 시연에는 단일 고확률 window는 차단하면서
감지력을 확보한 중간형 bitstream을 사용한다.

## 5. 수동 실행

```bash
cd ~/fall_detection
source yolovenv/bin/activate
export DISPLAY=:0
python real_time_fall_detection.py
```

systemd user service가 실행 중이면 카메라가 이미 점유되므로 수동 실행 전에 service를 중지한다.

```bash
systemctl --user stop fall-detection.service
```

## 6. 자동 실행

```bash
mkdir -p ~/.config/systemd/user
cp fall-detection.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now fall-detection.service
```

## 7. 상태 확인

```bash
systemctl --user status fall-detection.service
journalctl --user -u fall-detection.service -f
```

## 8. Reset

- Raspberry Pi/TCN 상태: desktop Reset 또는 `reset_alert.flag`
- FPGA alert latch: Basys3 BTNC
