import time
import os
import struct
import subprocess
import queue
import threading
from datetime import datetime
from pathlib import Path
import cv2
import requests
import numpy as np
from collections import deque

from picamera2 import Picamera2
from ultralytics import YOLO

import torch
import torch.nn as nn
import serial


BASE_DIR = Path(__file__).resolve().parent


def env_flag(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


# Model files are intentionally not committed. Put them in software/models or
# override these paths with environment variables.
YOLO_MODEL_PATH = os.getenv(
    "YOLO_MODEL_PATH",
    str(BASE_DIR / "models" / "yolo11n_blur_pose_best.pt"),
)
TCN_MODEL_PATH = os.getenv(
    "TCN_MODEL_PATH",
    str(BASE_DIR / "models" / "best_tcn_videowise.pt"),
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# TCN ?숈뒿 湲곗?
SEQ_LEN = 100
NUM_KEYPOINTS = 17
INPUT_DIM = 51

EARLY_MIN_LEN = 40

# YOLO keypoint 議곌굔
CONF_TH = 0.30
MIN_VALID_KPTS = 8

# TCN ?먮떒 湲곗?
FALL_TH = 0.70

# FALL ?뺤젙 議곌굔
FALL_CONFIRM_COUNT = 2

# ?щ엺?????≫엳???곹깭 泥섎━
NO_PERSON_RESET_SEC = 3.0
NO_PERSON_EVAL_SEC = 0.8

# ?몃? 由ъ뀑 ?뚯씪
RESET_FLAG_PATH = str(BASE_DIR / "reset_alert.flag")

# 移대찓??/ YOLO ?ㅼ젙
CAM_WIDTH = 320
CAM_HEIGHT = 240
CAM_FPS = int(os.getenv("CAM_FPS", "10"))
YOLO_IMGSZ = 320
YOLO_CONF = 0.25

# Privacy blur ?ㅼ젙
SHOW_BLUR = True
PRIVACY_BLUR_KSIZE = 11

# YOLO skeleton ?ㅼ떆媛??쒖떆
DRAW_YOLO_OVERLAY = True

# ?붾㈃ ?쒖떆 ?ㅼ젙
# VNC?먯꽌 ?곸긽 ?꾩슱 嫄곕㈃ True
# SSH留???嫄곕㈃ False 沅뚯옣
SHOW_PREVIEW = env_flag("SHOW_PREVIEW", True)

# ?붾㈃ 援ъ꽦
PANEL_WIDTH = 310
FONT = cv2.FONT_HERSHEY_SIMPLEX


# ============================================================
# 2. Telegram ?ㅼ젙
# ============================================================

# Direct Ethernet operation does not guarantee Internet/DNS access.
# Keep all Telegram HTTP calls disabled so they cannot stall the video loop.
SEND_TELEGRAM = env_flag("SEND_TELEGRAM", False)
SEND_SNAPSHOT = env_flag("SEND_SNAPSHOT", False)
SNAPSHOT_PATH = str(BASE_DIR / "fall_alert_snapshot_tcn_fpga_blurred.jpg")

# ?ш린????Telegram BotFather ?좏겙??吏곸젒 ?ｌ뼱
# ?? TELEGRAM_BOT_TOKEN = "1234567890:AAxxxxxxxxxxxxxxxxxxxxxxxx"
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")

# ??chat id
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

TELEGRAM_TIMEOUT = 3.0
TELEGRAM_PHOTO_TIMEOUT = 5.0

# Telegram reset polling
TELEGRAM_RESET_ENABLED = env_flag("TELEGRAM_RESET_ENABLED", False)
TELEGRAM_CHECK_INTERVAL = 5.0

# ?쒖옉????Telegram 硫붿떆吏 ?꾩넚
SEND_START_MESSAGE = env_flag("SEND_START_MESSAGE", False)


# ============================================================
# 3. FPGA UART ?ㅼ젙
# Pi TX GPIO14 pin8  -> Basys3 JA1 FPGA RX
# Pi RX GPIO15 pin10 <- Basys3 JA2 FPGA TX
# Pi GND             -> Basys3 JA GND
# UART: 115200, 8N1
# ============================================================

UART_ENABLED = env_flag("UART_ENABLED", True)
UART_PORT = os.getenv("UART_PORT", "/dev/serial0")
UART_BAUDRATE = int(os.getenv("UART_BAUDRATE", "115200"))

# FPGA GRU input format.
# The trained GRU and RTL consume 20 frames per inference. Raspberry Pi keeps
# a longer 100-frame history and continuously sends the newest 20-frame window.
GRU_SEQ_LEN = 20
GRU_HISTORY_LEN = 100
GRU_INPUT_DIM = 51
GRU_SEND_STRIDE = 5
GRU_WINDOWS_PER_CHUNK = (
    (GRU_HISTORY_LEN - GRU_SEQ_LEN) // GRU_SEND_STRIDE
) + 1
GRU_INPUT_SCALE = 64.0
GRU_INPUT_MIN = -(1 << 17)
GRU_INPUT_MAX = (1 << 17) - 1
GRU_FALL_THRESHOLD = 0.90

# FPGA濡?蹂대궪 9媛?keypoint
# COCO index:
# 0 nose
# 5 left_shoulder
# 6 right_shoulder
# 11 left_hip
# 12 right_hip
# 13 left_knee
# 14 right_knee
# 15 left_ankle
# 16 right_ankle
print("DEVICE:", DEVICE)


# ============================================================
# 4. FPGA UART ?대옒??# ============================================================

class FPGAUart:
    """Asynchronous UART link for the fixed-point FPGA GRU."""

    def __init__(self, port="/dev/serial0", baudrate=115200, timeout=0.05):
        self.ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=timeout
        )
        # Preserve every overlapping window in one 100-frame GRU chunk.
        self.tx_queue = queue.Queue(maxsize=GRU_WINDOWS_PER_CHUNK)
        self.rx_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.worker = threading.Thread(
            target=self._worker_loop,
            name="fpga-gru-uart",
            daemon=True
        )
        self.worker.start()

    @staticmethod
    def xor_checksum(data: bytes) -> int:
        c = 0
        for b in data:
            c ^= b
        return c & 0xFF

    @staticmethod
    def build_gru_packet(sequence_id, sequence):
        values = np.asarray(sequence, dtype=np.float32).reshape(
            GRU_SEQ_LEN,
            GRU_INPUT_DIM
        )
        quantized = np.clip(
            np.rint(values * GRU_INPUT_SCALE),
            GRU_INPUT_MIN,
            GRU_INPUT_MAX
        ).astype(np.int32).reshape(-1)

        payload = bytearray(struct.pack("<H", sequence_id & 0xFFFF))
        for value in quantized:
            raw = int(value) & 0x3FFFF
            payload.extend((
                raw & 0xFF,
                (raw >> 8) & 0xFF,
                (raw >> 16) & 0x03
            ))

        checksum = FPGAUart.xor_checksum(payload)
        return bytes((0xA5, 0x5A)) + bytes(payload) + bytes((checksum,))

    def submit_gru_sequence(self, sequence_id, sequence):
        if len(sequence) != GRU_SEQ_LEN:
            return False

        packet = self.build_gru_packet(sequence_id, sequence)
        try:
            self.tx_queue.put_nowait((sequence_id & 0xFFFF, packet))
            return True
        except queue.Full:
            return False

    def _read_exact(self, size, deadline):
        data = bytearray()
        while len(data) < size and time.monotonic() < deadline:
            chunk = self.ser.read(size - len(data))
            if chunk:
                data.extend(chunk)
        return bytes(data)

    def _read_gru_response(self, timeout=1.5):
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            first = self.ser.read(1)
            if not first or first[0] != 0xF1:
                continue

            tail = self._read_exact(6, deadline)
            if len(tail) != 6:
                break

            packet = first + tail
            if packet[6] != self.xor_checksum(packet[:6]):
                continue

            sequence_id, probability_q15 = struct.unpack("<HH", packet[1:5])
            return {
                "sequence_id": int(sequence_id),
                "probability_q15": int(probability_q15),
                "probability": float(probability_q15) / 32768.0,
                "fall": bool(packet[5] & 0x01),
                "raw": packet.hex(" ")
            }

        raise TimeoutError("FPGA GRU response timeout")

    def _worker_loop(self):
        while not self.stop_event.is_set():
            try:
                _, packet = self.tx_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                self.ser.reset_input_buffer()
                self.ser.write(packet)
                self.ser.flush()
                self.rx_queue.put(self._read_gru_response())
            except Exception as e:
                self.rx_queue.put({
                    "error": str(e),
                    "fall": False,
                    "raw": "UART_ERROR"
                })
            finally:
                self.tx_queue.task_done()

    def read_gru_result(self):
        try:
            return self.rx_queue.get_nowait()
        except queue.Empty:
            return None

    def clear(self):
        while True:
            try:
                self.tx_queue.get_nowait()
                self.tx_queue.task_done()
            except queue.Empty:
                break

        while True:
            try:
                self.rx_queue.get_nowait()
            except queue.Empty:
                break

        self.ser.reset_input_buffer()

    def close(self):
        self.stop_event.set()
        if self.worker.is_alive():
            self.worker.join(timeout=1.0)
        try:
            self.ser.close()
        except Exception:
            pass


# ============================================================
# 5. TCN 紐⑤뜽 ?뺤쓽
# PC?먯꽌 ?숈뒿??TCN 援ъ“? ?숈씪
# ============================================================

class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, dropout=0.2):
        super().__init__()

        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation
        )
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            in_channels=out_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation
        )
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = (
            nn.Conv1d(in_ch, out_ch, kernel_size=1)
            if in_ch != out_ch
            else None
        )

        self.relu = nn.ReLU()

    def chomp(self, x, chomp_size):
        if chomp_size == 0:
            return x

        return x[:, :, :-chomp_size]

    def forward(self, x):
        trim1 = self.conv1.padding[0]

        out = self.conv1(x)
        out = self.chomp(out, trim1)
        out = self.relu1(out)
        out = self.dropout1(out)

        trim2 = self.conv2.padding[0]

        out = self.conv2(out)
        out = self.chomp(out, trim2)
        out = self.relu2(out)
        out = self.dropout2(out)

        residual = x if self.downsample is None else self.downsample(x)

        return self.relu(out + residual)


class FallTCN(nn.Module):
    def __init__(self, input_dim=51, num_classes=2, channels=None, dropout=0.2):
        super().__init__()

        if channels is None:
            channels = [64, 128, 128]

        layers = []
        in_ch = input_dim

        for i, out_ch in enumerate(channels):
            dilation = 2 ** i

            layers.append(
                TemporalBlock(
                    in_ch=in_ch,
                    out_ch=out_ch,
                    kernel_size=3,
                    dilation=dilation,
                    dropout=dropout
                )
            )

            in_ch = out_ch

        self.network = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels[-1], num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.network(x)
        x = self.pool(x)
        x = x.squeeze(-1)
        out = self.fc(x)

        return out


# ============================================================
# 6. 紐⑤뜽 濡쒕뱶
# ============================================================

def load_tcn_model(path):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)

    channels = ckpt.get("channels", [64, 128, 128])
    dropout = ckpt.get("dropout", 0.2)

    model = FallTCN(
        input_dim=INPUT_DIM,
        num_classes=2,
        channels=channels,
        dropout=dropout
    )

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    print("TCN 濡쒕뱶 ?꾨즺")
    print("best_f1:", ckpt.get("best_f1"))
    print("best_recall:", ckpt.get("best_recall"))
    print("seq_len:", ckpt.get("seq_len", "NO seq_len key"))
    print("channels:", channels)

    return model


print("YOLO_MODEL_PATH:", YOLO_MODEL_PATH)
print("TCN_MODEL_PATH:", TCN_MODEL_PATH)

pose_model = YOLO(YOLO_MODEL_PATH)
print("YOLO Pose 濡쒕뱶 ?꾨즺")

tcn_model = load_tcn_model(TCN_MODEL_PATH)


# ============================================================
# 7. ?좏떥 ?⑥닔
# ============================================================

def now_time_string():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def apply_privacy_blur(frame):
    k = PRIVACY_BLUR_KSIZE

    if k % 2 == 0:
        k += 1

    return cv2.GaussianBlur(frame, (k, k), 0)


def telegram_ready():
    if not SEND_TELEGRAM:
        return False

    if TELEGRAM_BOT_TOKEN.strip() == "":
        return False

    if TELEGRAM_CHAT_ID.strip() == "":
        return False

    return True


# ============================================================
# 8. Telegram ?⑥닔
# ============================================================

def send_telegram_message(message):
    if not telegram_ready():
        print("Telegram not ready")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message
    }

    try:
        response = requests.post(
            url,
            data=payload,
            timeout=TELEGRAM_TIMEOUT
        )

        if response.status_code == 200:
            print("Telegram message sent")
            return True

        print("Telegram message failed:", response.status_code, response.text)
        return False

    except Exception as e:
        print("Telegram message error:", e)
        return False


def send_telegram_photo(image_path, caption=""):
    if not telegram_ready():
        print("Telegram not ready")
        return False

    if not SEND_SNAPSHOT:
        return False

    if not os.path.exists(image_path):
        print("Snapshot file does not exist:", image_path)
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"

    try:
        with open(image_path, "rb") as photo:
            files = {
                "photo": photo
            }

            data = {
                "chat_id": TELEGRAM_CHAT_ID,
                "caption": caption
            }

            response = requests.post(
                url,
                data=data,
                files=files,
                timeout=TELEGRAM_PHOTO_TIMEOUT
            )

        if response.status_code == 200:
            print("Telegram photo sent")
            return True

        print("Telegram photo failed:", response.status_code, response.text)
        return False

    except Exception as e:
        print("Telegram photo error:", e)
        return False


def telegram_get_updates(offset=None):
    if not TELEGRAM_RESET_ENABLED:
        return []

    if not telegram_ready():
        return []

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"

    params = {
        "timeout": 0
    }

    if offset is not None:
        params["offset"] = offset

    try:
        response = requests.get(
            url,
            params=params,
            timeout=TELEGRAM_TIMEOUT
        )

        data = response.json()

        return data.get("result", [])

    except Exception as e:
        print("Telegram getUpdates error:", e)
        return []


def trigger_alert_protocol(snapshot_path=None, source="SYSTEM", fpga_vector=0, raw=""):
    alert_time_str = now_time_string()

    print("====================================")
    print("FALL ALERT TRIGGERED - TCN + FPGA")
    print("source:", source)
    print("fpga_vector:", fpga_vector)
    print("raw:", raw)
    print("time:", alert_time_str)
    print("====================================")

    message = (
        "?슚 FALL ALERT\n"
        f"Source: {source}\n"
        f"Time: {alert_time_str}"
    )

    send_telegram_message(message)


def show_local_emergency_alert(alert_time_str):
    global local_alert_process

    if local_alert_process is not None and local_alert_process.poll() is None:
        return

    env = os.environ.copy()
    env.setdefault("DISPLAY", ":0")
    env.setdefault("WAYLAND_DISPLAY", "wayland-0")
    env.setdefault("XDG_RUNTIME_DIR", "/run/user/1000")
    env.setdefault(
        "DBUS_SESSION_BUS_ADDRESS",
        "unix:path=/run/user/1000/bus"
    )

    message = (
        "<span size='xx-large' weight='bold' foreground='red'>"
        "FALL DETECTED"
        "</span>\n\n"
        "<span size='x-large' weight='bold'>CALL POLICE</span>\n\n"
        "Both FPGA and TCN confirmed a fall.\n"
        f"Time: {alert_time_str}"
    )

    try:
        local_alert_process = subprocess.Popen(
            [
                "/usr/bin/zenity",
                "--warning",
                "--title=Fall Detection Emergency",
                f"--text={message}",
                "--width=520",
                "--height=260",
                "--no-wrap",
            ],
            env=env,
        )
        print("LOCAL EMERGENCY POPUP DISPLAYED")
    except Exception as e:
        print("Local emergency popup error:", e)


def trigger_combined_alert_if_needed(snapshot_path=None):
    global combined_alert_sent

    if combined_alert_sent:
        return

    if not (fpga_alert_sent and tcn_alert_sent):
        return

    combined_alert_sent = True
    alert_time_str = now_time_string()

    print("====================================")
    print("CALL POLICE - BOTH FPGA AND TCN CONFIRMED")
    print("time:", alert_time_str)
    print("====================================")

    show_local_emergency_alert(alert_time_str)

    send_telegram_message(
        "CALL POLICE\n"
        "Both FPGA GRU and Raspberry Pi TCN confirmed a fall.\n"
        f"Time: {alert_time_str}"
    )

    if snapshot_path is not None:
        send_telegram_photo(
            snapshot_path,
            caption=(
                "CALL POLICE\n"
                "Both FPGA GRU and Raspberry Pi TCN confirmed a fall.\n"
                f"Time: {alert_time_str}"
            )
        )


# ============================================================
# 9. YOLO Pose ?꾩쿂由?# ============================================================

def select_largest_person(result):
    if result.boxes is None or len(result.boxes) == 0:
        return None

    boxes = result.boxes.xyxy.cpu().numpy()

    areas = []

    for box in boxes:
        x1, y1, x2, y2 = box
        area = max(0, x2 - x1) * max(0, y2 - y1)
        areas.append(area)

    if len(areas) == 0:
        return None

    return int(np.argmax(areas))


def normalize_keypoints_for_tcn(kpts, width, height):
    out = np.zeros((NUM_KEYPOINTS, 3), dtype=np.float32)

    if kpts is None:
        return out

    out[:, 0] = kpts[:, 0] / max(width, 1)
    out[:, 1] = kpts[:, 1] / max(height, 1)
    out[:, 2] = kpts[:, 2]

    out[:, 0] = np.clip(out[:, 0], 0.0, 1.0)
    out[:, 1] = np.clip(out[:, 1], 0.0, 1.0)
    out[:, 2] = np.clip(out[:, 2], 0.0, 1.0)

    return out


def normalize_keypoints_for_gru(kpts):
    """Match the preprocessing used to train fall2d_gru.pt."""
    out = np.asarray(kpts, dtype=np.float32).copy()
    xy = out[:, :2]
    confidence = out[:, 2:3]

    left_hip = out[11]
    right_hip = out[12]
    if left_hip[2] > 0 and right_hip[2] > 0:
        center = (left_hip[:2] + right_hip[:2]) / 2.0
    else:
        valid = out[:, 2] > 0
        center = xy[valid].mean(axis=0) if valid.any() else np.zeros(2, dtype=np.float32)

    left_shoulder = out[5]
    right_shoulder = out[6]
    if left_shoulder[2] > 0 and right_shoulder[2] > 0:
        scale = np.linalg.norm(left_shoulder[:2] - right_shoulder[:2])
    elif left_hip[2] > 0 and right_hip[2] > 0:
        scale = np.linalg.norm(left_hip[:2] - right_hip[:2])
    else:
        scale = 1.0

    if scale < 1e-6:
        scale = 1.0

    out[:, :2] = (xy - center) / scale
    out[:, 2:3] = confidence
    return out


def draw_skeleton(frame, kpts):
    pairs = [
        (5, 6),
        (5, 7), (7, 9),
        (6, 8), (8, 10),
        (5, 11), (6, 12),
        (11, 12),
        (11, 13), (13, 15),
        (12, 14), (14, 16)
    ]

    for i, j in pairs:
        if kpts[i, 2] > 0 and kpts[j, 2] > 0:
            x1, y1 = int(kpts[i, 0]), int(kpts[i, 1])
            x2, y2 = int(kpts[j, 0]), int(kpts[j, 1])
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

    for i in range(17):
        if kpts[i, 2] > 0:
            x, y = int(kpts[i, 0]), int(kpts[i, 1])
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

    return frame


# ============================================================
# 10. TCN 異붾줎
# ============================================================

def make_tcn_input_from_buffer(seq_buffer, pad_to_len=SEQ_LEN):
    seq = list(seq_buffer)

    if len(seq) == 0:
        return None

    if len(seq) >= pad_to_len:
        seq = seq[-pad_to_len:]
    else:
        last = seq[-1]
        pad_count = pad_to_len - len(seq)

        for _ in range(pad_count):
            seq.append(last.copy())

    arr = np.stack(seq, axis=0).astype(np.float32)
    arr = arr.reshape(pad_to_len, INPUT_DIM)

    return arr


def predict_from_sequence(seq_buffer):
    x_np = make_tcn_input_from_buffer(seq_buffer, SEQ_LEN)

    if x_np is None:
        return 0.0, False

    x_tensor = torch.tensor(
        x_np,
        dtype=torch.float32
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = tcn_model(x_tensor)
        probs = torch.softmax(logits, dim=1)
        fall_prob = probs[0, 1].item()

    final_is_fall = fall_prob >= FALL_TH

    return float(fall_prob), final_is_fall


# ============================================================
# 11. ?붾㈃ canvas ?앹꽦
# ============================================================

def make_display_canvas(
    display_frame,
    last_prob,
    valid_kpts,
    buffer_len,
    gru_buffer_len,
    gru_windows_submitted,
    fps,
    fall_confirm,
    uart_ok,
    tcn_fall,
    fpga_fall,
    combined_fall,
    fpga_probability
):
    h, w = display_frame.shape[:2]

    canvas = np.zeros((h, w + PANEL_WIDTH, 3), dtype=np.uint8)
    canvas[:, :w] = display_frame

    panel = canvas[:, w:w + PANEL_WIDTH]
    panel[:] = (25, 25, 25)

    true_color = (0, 0, 255)
    false_color = (0, 255, 0)
    info_color = (220, 220, 220)

    if combined_fall:
        decision_text = "CALL POLICE"
        decision_color = true_color
    elif fpga_fall:
        decision_text = "WAITING FOR PI5 TCN"
        decision_color = (0, 255, 255)
    elif tcn_fall:
        decision_text = "WAITING FOR FPGA GRU"
        decision_color = (0, 255, 255)
    else:
        decision_text = "MONITORING"
        decision_color = false_color

    x0 = w + 10
    y = 18
    line = 14

    cv2.putText(canvas, "Fall Decision Pipeline", (x0, y), FONT, 0.45, (255, 255, 255), 1)
    y += line

    cv2.putText(canvas, "Independent model decisions", (x0, y), FONT, 0.30, info_color, 1)
    y += line + 5

    cv2.putText(
        canvas,
        f"FPGA GRU FALL : {str(bool(fpga_fall)).upper()}",
        (x0, y),
        FONT,
        0.38,
        true_color if fpga_fall else false_color,
        1
    )
    y += line
    cv2.putText(
        canvas,
        f"  probability={fpga_probability:.2f}  windows={gru_windows_submitted}/{GRU_WINDOWS_PER_CHUNK}",
        (x0, y),
        FONT,
        0.27,
        info_color,
        1
    )
    y += line + 4

    cv2.putText(
        canvas,
        f"Pi5 TCN FALL : {str(bool(tcn_fall)).upper()}",
        (x0, y),
        FONT,
        0.38,
        true_color if tcn_fall else false_color,
        1
    )
    y += line
    cv2.putText(
        canvas,
        f"  probability={last_prob:.2f}  confirm={fall_confirm}/{FALL_CONFIRM_COUNT}",
        (x0, y),
        FONT,
        0.27,
        info_color,
        1
    )
    y += line + 5

    cv2.line(canvas, (w + 8, y), (w + PANEL_WIDTH - 8, y), (90, 90, 90), 1)
    y += line
    fusion_text = (
        f"{str(bool(fpga_fall)).upper()} AND "
        f"{str(bool(tcn_fall)).upper()} -> "
        f"{str(bool(combined_fall)).upper()}"
    )
    cv2.putText(canvas, "Fusion: FPGA AND TCN", (x0, y), FONT, 0.34, (255, 255, 255), 1)
    y += line
    cv2.putText(
        canvas,
        fusion_text,
        (x0, y),
        FONT,
        0.35,
        true_color if combined_fall else (0, 255, 255),
        1
    )
    y += line + 4

    cv2.putText(canvas, f"Decision: {decision_text}", (x0, y), FONT, 0.36, decision_color, 1)
    y += line + 4

    cv2.line(canvas, (w + 8, y), (w + PANEL_WIDTH - 8, y), (90, 90, 90), 1)
    y += line
    cv2.putText(
        canvas,
        f"TCN {buffer_len}/{SEQ_LEN}  GRU {gru_buffer_len}/{GRU_HISTORY_LEN}",
        (x0, y),
        FONT,
        0.29,
        info_color,
        1
    )
    y += line
    cv2.putText(
        canvas,
        f"kpts={valid_kpts}  FPS={fps:.1f}  UART={uart_ok}",
        (x0, y),
        FONT,
        0.29,
        info_color,
        1
    )

    return canvas


# ============================================================
# 12. Picamera2 ?쒖옉
# ============================================================

picam2 = Picamera2()

camera_config = picam2.create_preview_configuration(
    main={
        "size": (CAM_WIDTH, CAM_HEIGHT),
        "format": "RGB888"
    }
)

picam2.configure(camera_config)
picam2.start()

time.sleep(1)

print("Picamera2 ?쒖옉 ?꾨즺")


# ============================================================
# 13. UART ?쒖옉
# ============================================================

uart = None
uart_ok = False

if UART_ENABLED:
    try:
        uart = FPGAUart(port=UART_PORT, baudrate=UART_BAUDRATE)
        uart_ok = True
        print(f"FPGA UART ?쒖옉 ?꾨즺: {UART_PORT}, {UART_BAUDRATE} baud")
    except Exception as e:
        uart = None
        uart_ok = False
        print("FPGA UART ?쒖옉 ?ㅽ뙣:", e)


# ============================================================
# 14. Buffer / ?곹깭 蹂??# ============================================================

seq_buffer = deque(maxlen=SEQ_LEN)
gru_buffer = deque(maxlen=GRU_HISTORY_LEN)
gru_valid_frame_count = 0
gru_windows_submitted = 0
gru_sequence_id = 0

alert_triggered = False
alert_time = None
fpga_alert_sent = False
tcn_alert_sent = False
combined_alert_sent = False
local_alert_process = None

last_result_text = "NONE"
current_result_text = "COLLECTING"

last_prob = 0.0
last_predict_reason = "NONE"

no_person_start_time = None

prev_time = time.time()
fps = 0.0
last_pipeline_log_time = 0.0
last_pipeline_signature = None

last_update_id = None
last_telegram_check_time = 0.0

fall_confirm_counter = 0

frame_seq = 0
fpga_fall = False
fpga_alert_vector = 0
fpga_raw = "NONE"
last_fpga_alert_time = None
last_fpga_probability = 0.0
last_fpga_sequence_id = None
last_fpga_uart_error = ""

print("====================================")
print("?ㅼ떆媛??숈긽 媛먯? ?쒖옉 - TCN + FPGA GRU UART + Telegram Version")
print(
    f"TCN_SEQ_LEN={SEQ_LEN}, "
    f"GRU_HISTORY_LEN={GRU_HISTORY_LEN}, "
    f"GRU_WINDOW_LEN={GRU_SEQ_LEN}, "
    f"GRU_STRIDE={GRU_SEND_STRIDE}, "
    f"GRU_WINDOWS={GRU_WINDOWS_PER_CHUNK}"
)
print("SHOW_PREVIEW:", SHOW_PREVIEW)
print("UART_OK:", uart_ok)
print("TG_READY:", telegram_ready())
print("====================================")

if SEND_START_MESSAGE:
    start_time_str = now_time_string()

    send_telegram_message(
        "??Fall detection started\n"
        f"Time: {start_time_str}"
    )


# ============================================================
# 15. Reset ?⑥닔
# ============================================================

def reset_alert_state():
    global alert_triggered
    global alert_time
    global fpga_alert_sent
    global tcn_alert_sent
    global combined_alert_sent
    global local_alert_process
    global last_result_text
    global current_result_text
    global last_prob
    global no_person_start_time
    global last_predict_reason
    global fall_confirm_counter
    global fpga_fall
    global fpga_alert_vector
    global fpga_raw
    global last_fpga_alert_time
    global gru_valid_frame_count
    global gru_windows_submitted
    global gru_sequence_id
    global last_fpga_probability
    global last_fpga_sequence_id
    global last_fpga_uart_error

    alert_triggered = False
    alert_time = None
    fpga_alert_sent = False
    tcn_alert_sent = False
    combined_alert_sent = False

    if local_alert_process is not None and local_alert_process.poll() is None:
        local_alert_process.terminate()
    local_alert_process = None
    seq_buffer.clear()
    gru_buffer.clear()
    gru_valid_frame_count = 0
    gru_windows_submitted = 0
    gru_sequence_id = 0
    if uart is not None:
        uart.clear()

    no_person_start_time = None

    last_result_text = "RESET"
    current_result_text = "COLLECTING"

    last_prob = 0.0
    last_predict_reason = "RESET"
    fall_confirm_counter = 0

    fpga_fall = False
    fpga_alert_vector = 0
    fpga_raw = "RESET"
    last_fpga_alert_time = None
    last_fpga_probability = 0.0
    last_fpga_sequence_id = None
    last_fpga_uart_error = ""

    print("\nALERT RESET / TCN+GRU BUFFER CLEARED / FPGA STATE CLEARED")


def append_gru_frame(gru_frame):
    global gru_valid_frame_count
    global gru_windows_submitted
    global gru_sequence_id

    gru_buffer.append(gru_frame)
    gru_valid_frame_count += 1

    should_submit = (
        uart_ok
        and uart is not None
        and len(gru_buffer) >= GRU_SEQ_LEN
        and (gru_valid_frame_count % GRU_SEND_STRIDE) == 0
    )

    if should_submit:
        submitted = uart.submit_gru_sequence(
            sequence_id=gru_sequence_id,
            sequence=list(gru_buffer)[-GRU_SEQ_LEN:]
        )
        if submitted:
            gru_sequence_id = (gru_sequence_id + 1) & 0xFFFF
            gru_windows_submitted += 1

    if gru_valid_frame_count >= GRU_HISTORY_LEN:
        if gru_windows_submitted != GRU_WINDOWS_PER_CHUNK:
            print(
                "\n[FPGA GRU CHUNK WARNING]",
                f"submitted={gru_windows_submitted}/"
                f"{GRU_WINDOWS_PER_CHUNK}"
            )

        gru_buffer.clear()
        gru_valid_frame_count = 0
        gru_windows_submitted = 0
        print("\n[FPGA GRU CHUNK COMPLETE] buffer reset to 0/100")


# ============================================================
# 16. Telegram reset ?뺤씤
# ============================================================

def check_telegram_reset():
    global last_update_id

    updates = telegram_get_updates(
        last_update_id + 1 if last_update_id is not None else None
    )

    reset_requested = False

    for update in updates:
        last_update_id = update.get("update_id", last_update_id)

        message = update.get("message", {})
        text = message.get("text", "")
        cmd = text.strip().lower()

        if cmd in ["/reset", "reset"]:
            reset_requested = True

    return reset_requested


# ============================================================
# 17. TCN ?먮퀎 ?ㅽ뻾 怨듯넻 ?⑥닔
# ============================================================

def run_tcn_prediction(predict_reason, display_frame, current_time):
    global alert_triggered
    global alert_time
    global tcn_alert_sent
    global current_result_text
    global last_result_text
    global last_prob
    global last_predict_reason
    global fall_confirm_counter

    fall_prob, final_is_fall = predict_from_sequence(seq_buffer)

    last_prob = fall_prob
    last_predict_reason = predict_reason

    if final_is_fall:
        fall_confirm_counter += 1
    else:
        fall_confirm_counter = 0

    if fall_confirm_counter >= FALL_CONFIRM_COUNT:
        current_result_text = "FALL"
        last_result_text = "FALL"

        if not tcn_alert_sent:
            tcn_alert_sent = True

            if not alert_triggered:
                alert_time = current_time

            alert_triggered = True

            cv2.imwrite(SNAPSHOT_PATH, display_frame)

            trigger_alert_protocol(
                snapshot_path=SNAPSHOT_PATH,
                source="PI_TCN",
                fpga_vector=fpga_alert_vector,
                raw=fpga_raw
            )

        trigger_combined_alert_if_needed(snapshot_path=SNAPSHOT_PATH)

    else:
        current_result_text = "NO FALL"
        last_result_text = "NO FALL"

    seq_buffer.clear()


# ============================================================
# 18. FPGA alert 泥섎━ ?⑥닔
# ============================================================

def check_fpga_result(display_frame, current_time):
    global fpga_fall
    global fpga_alert_vector
    global fpga_raw
    global last_fpga_alert_time
    global last_fpga_probability
    global last_fpga_sequence_id
    global last_fpga_uart_error
    global alert_triggered
    global alert_time
    global fpga_alert_sent
    global current_result_text
    global last_result_text
    global last_predict_reason

    if not uart_ok or uart is None:
        return

    try:
        result = uart.read_gru_result()
    except Exception as e:
        print("\nFPGA GRU UART read error:", e)
        return

    if result is None:
        return

    if "error" in result:
        last_fpga_uart_error = result["error"]
        fpga_raw = result.get("raw", "UART_ERROR")
        last_predict_reason = "FPGA_GRU_UART_ERROR"
        print("\n[FPGA GRU UART ERROR]", last_fpga_uart_error)
        return

    last_fpga_uart_error = ""
    last_fpga_probability = float(result.get("probability", 0.0))
    last_fpga_sequence_id = result.get("sequence_id")
    fpga_alert_vector = 1 if result.get("fall", False) else 0
    fpga_raw = result.get("raw", "")
    last_fpga_alert_time = current_time
    last_predict_reason = "FPGA_GRU_RESULT"

    print(
        "\n[FPGA GRU RESULT]",
        f"seq={last_fpga_sequence_id}",
        f"prob={last_fpga_probability:.6f}",
        f"fall={fpga_alert_vector}",
        fpga_raw
    )

    if not result.get("fall", False):
        if not fpga_alert_sent:
            fpga_fall = False
        return

    fpga_fall = True
    current_result_text = "FPGA FALL"
    last_result_text = "FPGA FALL"
    last_predict_reason = "FPGA_GRU_FALL"

    if not fpga_alert_sent:
        fpga_alert_sent = True

        if not alert_triggered:
            alert_time = current_time

        alert_triggered = True

        cv2.imwrite(SNAPSHOT_PATH, display_frame)

        trigger_alert_protocol(
            snapshot_path=SNAPSHOT_PATH,
            source="FPGA_GRU",
            fpga_vector=fpga_alert_vector,
            raw=fpga_raw
        )

    trigger_combined_alert_if_needed(snapshot_path=SNAPSHOT_PATH)


# ============================================================
# 19. ?ㅼ떆媛?猷⑦봽
# ============================================================

try:
    while True:
        frame_rgb = picam2.capture_array()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        yolo_frame = frame_bgr.copy()

        if SHOW_BLUR:
            display_frame = apply_privacy_blur(frame_bgr.copy())
        else:
            display_frame = frame_bgr.copy()

        results = pose_model.predict(
            source=yolo_frame,
            imgsz=YOLO_IMGSZ,
            conf=YOLO_CONF,
            verbose=False
        )

        result = results[0]
        kpts_out = np.zeros((17, 3), dtype=np.float32)

        if result.keypoints is not None:
            person_idx = select_largest_person(result)

            if person_idx is not None:
                kpts = result.keypoints.data.cpu().numpy()

                if person_idx < len(kpts):
                    selected = kpts[person_idx]

                    for j in range(17):
                        x, y, c = selected[j]

                        if c >= CONF_TH:
                            kpts_out[j] = [x, y, c]

        valid_kpts = int((kpts_out[:, 2] > CONF_TH).sum())

        if DRAW_YOLO_OVERLAY:
            display_frame = draw_skeleton(display_frame, kpts_out)

        current_time = time.time()

        check_fpga_result(
            display_frame=display_frame,
            current_time=current_time
        )

        if valid_kpts >= MIN_VALID_KPTS:
            no_person_start_time = None

            h, w = yolo_frame.shape[:2]
            kpts_norm = normalize_keypoints_for_tcn(kpts_out, w, h)
            seq_buffer.append(kpts_norm)

            gru_norm = normalize_keypoints_for_gru(kpts_out)
            append_gru_frame(gru_norm)

        else:
            if no_person_start_time is None:
                no_person_start_time = current_time

            no_person_elapsed = current_time - no_person_start_time

            if (
                len(seq_buffer) >= EARLY_MIN_LEN
                and no_person_elapsed >= NO_PERSON_EVAL_SEC
            ):
                run_tcn_prediction(
                    predict_reason="NO_PERSON_EARLY",
                    display_frame=display_frame,
                    current_time=current_time
                )

                no_person_start_time = current_time

            elif (
                not alert_triggered
                and no_person_elapsed >= NO_PERSON_RESET_SEC
            ):
                seq_buffer.clear()
                gru_buffer.clear()
                gru_valid_frame_count = 0
                gru_windows_submitted = 0
                current_result_text = "NO PERSON"
                last_result_text = "NO PERSON"
                last_prob = 0.0
                last_predict_reason = "NO_PERSON_RESET"
                fall_confirm_counter = 0

        if len(seq_buffer) >= SEQ_LEN:
            run_tcn_prediction(
                predict_reason="FULL_SEQ",
                display_frame=display_frame,
                current_time=current_time
            )

        if os.path.exists(RESET_FLAG_PATH):
            reset_alert_state()
            os.remove(RESET_FLAG_PATH)

            send_telegram_message(
                "Fall alert reset completed\n"
                f"Time: {now_time_string()}"
            )

        now_for_telegram = time.time()

        if telegram_ready() and (now_for_telegram - last_telegram_check_time >= TELEGRAM_CHECK_INTERVAL):
            last_telegram_check_time = now_for_telegram

            if check_telegram_reset():
                reset_alert_state()

                send_telegram_message(
                    "Fall alert reset completed\n"
                    f"Time: {now_time_string()}"
                )

        if combined_alert_sent:
            current_display_text = "CALL POLICE"

        elif alert_triggered and fpga_fall:
            current_display_text = "FPGA FALL"

        elif alert_triggered:
            current_display_text = "FALL"

        elif valid_kpts < MIN_VALID_KPTS:
            current_display_text = "NO PERSON"

        elif len(seq_buffer) < SEQ_LEN:
            current_display_text = "COLLECTING"

        else:
            current_display_text = current_result_text

        alert_text = "ALERT: TRIGGERED" if alert_triggered else "ALERT: NOT TRIGGERED"

        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        if alert_triggered and alert_time is not None:
            alert_elapsed = current_time - alert_time
        else:
            alert_elapsed = None

        canvas = make_display_canvas(
            display_frame=display_frame,
            last_prob=last_prob,
            valid_kpts=valid_kpts,
            buffer_len=len(seq_buffer),
            gru_buffer_len=len(gru_buffer),
            gru_windows_submitted=gru_windows_submitted,
            fps=fps,
            fall_confirm=fall_confirm_counter,
            uart_ok=uart_ok,
            tcn_fall=tcn_alert_sent,
            fpga_fall=fpga_alert_sent,
            combined_fall=bool(fpga_alert_sent and tcn_alert_sent),
            fpga_probability=last_fpga_probability
        )

        fpga_gru_fall_state = bool(fpga_alert_sent)
        pi5_tcn_fall_state = bool(tcn_alert_sent)
        fusion_fall_state = bool(fpga_gru_fall_state and pi5_tcn_fall_state)

        if fusion_fall_state:
            pipeline_action = "CALL POLICE"
        elif fpga_gru_fall_state:
            pipeline_action = "WAITING FOR PI5 TCN"
        elif pi5_tcn_fall_state:
            pipeline_action = "WAITING FOR FPGA GRU"
        else:
            pipeline_action = "MONITORING"

        pipeline_signature = (
            fpga_gru_fall_state,
            pi5_tcn_fall_state,
            fusion_fall_state,
            pipeline_action,
        )

        if (
            pipeline_signature != last_pipeline_signature
            or now - last_pipeline_log_time >= 1.0
        ):
            print(
                "\n[DECISION] "
                f"FPGA GRU FALL : {fpga_gru_fall_state} "
                f"(p={last_fpga_probability:.3f}, "
                f"windows={gru_windows_submitted}/{GRU_WINDOWS_PER_CHUNK}) | "
                f"Pi5 TCN FALL : {pi5_tcn_fall_state} "
                f"(p={last_prob:.3f}, "
                f"confirm={fall_confirm_counter}/{FALL_CONFIRM_COUNT})"
            )
            print(
                "[FUSION] "
                f"{fpga_gru_fall_state} AND {pi5_tcn_fall_state} "
                f"-> {fusion_fall_state} | ACTION: {pipeline_action} | "
                f"TCN={len(seq_buffer)}/{SEQ_LEN} "
                f"GRU={len(gru_buffer)}/{GRU_HISTORY_LEN} "
                f"kpts={valid_kpts} fps={fps:.2f}"
            )
            last_pipeline_signature = pipeline_signature
            last_pipeline_log_time = now

        if SHOW_PREVIEW:
            cv2.imshow("Fall Detection - Raspberry Pi TCN + FPGA GRU", canvas)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("r") or key == ord("R"):
                reset_alert_state()

                send_telegram_message(
                    "Fall alert reset completed\n"
                    f"Time: {now_time_string()}"
                )

        frame_seq = (frame_seq + 1) & 0xFFFF

except KeyboardInterrupt:
    print("\nKeyboardInterrupt 醫낅즺")


# ============================================================
# 20. 
# ============================================================

picam2.stop()

if uart is not None:
    uart.close()

if SHOW_PREVIEW:
    cv2.destroyAllWindows()

send_telegram_message(
    "Fall detection stopped\n"
    f"Time: {now_time_string()}"
)

print("\n醫낅즺")

