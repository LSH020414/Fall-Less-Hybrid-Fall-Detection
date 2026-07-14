import time
import os
from datetime import datetime
import cv2
import requests
import numpy as np
from collections import deque

from picamera2 import Picamera2
from ultralytics import YOLO

import torch
import torch.nn as nn


# ============================================================
# 1. 설정
# 원본 real_time_fall_detection.py 판단 구조 유지
# ============================================================

MODEL_PATH = "fall2d_gru.pt"
YOLO_PATH = "yolov8n-pose.pt"

# GRU 학습 기준: 10fps 기준 2초 = 20프레임
# 원본 유지
SEQ_LEN = 20

# 10초 단위 chunk 판별
# 원본 유지
CHUNK_LEN = 100
WINDOW_STRIDE = 5

# YOLO keypoint confidence
CONF_TH = 0.30
MIN_VALID_KPTS = 8

# 10초 chunk 판별 기준
# 원본 유지
FALL_TH = 0.90
MIN_FALL_WINDOWS = 5
AVG_FALL_TH = 0.75

# 움직임 조건
# 원본 유지
MOTION_TH = 0.06

# 사람이 안 잡히는 상태 처리
NO_PERSON_RESET_SEC = 3.0

# 추가:
# 사람이 갑자기 안 잡히기 시작했을 때,
# chunk가 100까지 안 차도 최소 20프레임 이상 있으면 조기 판별
NO_PERSON_EVAL_SEC = 0.8

# 외부 리셋 파일
RESET_FLAG_PATH = "reset_alert.flag"

# 카메라 / YOLO 설정
# 원본의 빠른 설정 유지
IMG_SIZE = 320
CAM_WIDTH = 320
CAM_HEIGHT = 240

# Privacy blur 설정
# YOLO 입력에는 원본 사용, 화면/스냅샷에만 블러 적용
SHOW_BLUR = True
PRIVACY_BLUR_KSIZE = 11

# YOLO skeleton 실시간 표시
# 블러된 화면 위에 skeleton 표시
DRAW_YOLO_OVERLAY = True

# 화면 구성
PANEL_WIDTH = 270
FONT = cv2.FONT_HERSHEY_SIMPLEX

# 텔레그램 설정
TELEGRAM_BOT_TOKEN = "8872957794:AAGHDmlWY0JqJwR_AMPE3gDCtYL856a8MP4"
TELEGRAM_CHAT_ID = "7629458731"

SEND_TELEGRAM = True
SEND_SNAPSHOT = True
SNAPSHOT_PATH = "fall_alert_snapshot_blurred.jpg"

# Telegram reset polling
TELEGRAM_RESET_ENABLED = True
TELEGRAM_CHECK_INTERVAL = 5.0
TELEGRAM_TIMEOUT = 0.5
TELEGRAM_PHOTO_TIMEOUT = 3.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("DEVICE:", DEVICE)


# ============================================================
# 2. GRU 모델 정의
# 원본 구조 유지
# ============================================================

class GRUModel(nn.Module):
    def __init__(self, input_size=51, hidden_size=64, num_layers=1, num_classes=2):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


# ============================================================
# 3. GRU 모델 로드
# ============================================================

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
config = checkpoint["config"]

gru_model = GRUModel(
    input_size=config["input_size"],
    hidden_size=config["hidden_size"],
    num_layers=config["num_layers"],
    num_classes=config["num_classes"]
).to(DEVICE)

gru_model.load_state_dict(checkpoint["model_state"])
gru_model.eval()

print("GRU 로드 완료")
print("GRU config:", config)


# ============================================================
# 4. YOLO Pose 모델 로드
# ============================================================

pose_model = YOLO(YOLO_PATH)

print("YOLO Pose 로드 완료")


# ============================================================
# 5. 유틸 함수
# ============================================================

def now_time_string():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def apply_privacy_blur(frame):
    k = PRIVACY_BLUR_KSIZE

    if k % 2 == 0:
        k += 1

    return cv2.GaussianBlur(frame, (k, k), 0)


# ============================================================
# 6. Skeleton 정규화
# 원본 유지
# ============================================================

LS, RS = 5, 6
LH, RH = 11, 12


def normalize_frame(kpts):
    kpts = kpts.copy().astype(np.float32)

    xy = kpts[:, :2]
    conf = kpts[:, 2:3]

    left_hip = kpts[LH]
    right_hip = kpts[RH]

    if left_hip[2] > 0 and right_hip[2] > 0:
        center = (left_hip[:2] + right_hip[:2]) / 2.0
    else:
        valid = kpts[:, 2] > 0

        if valid.sum() > 0:
            center = xy[valid].mean(axis=0)
        else:
            center = np.array([0.0, 0.0], dtype=np.float32)

    left_sh = kpts[LS]
    right_sh = kpts[RS]

    if left_sh[2] > 0 and right_sh[2] > 0:
        scale = np.linalg.norm(left_sh[:2] - right_sh[:2])
    elif left_hip[2] > 0 and right_hip[2] > 0:
        scale = np.linalg.norm(left_hip[:2] - right_hip[:2])
    else:
        scale = 1.0

    if scale < 1e-6:
        scale = 1.0

    kpts[:, :2] = (xy - center) / scale
    kpts[:, 2:3] = conf

    return kpts


# ============================================================
# 7. 가장 큰 사람 선택
# 원본 유지
# ============================================================

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


# ============================================================
# 8. Skeleton 그리기
# 블러된 display_frame 위에만 그림
# ============================================================

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
# 9. Motion score 계산
# 원본 유지
# ============================================================

def compute_motion_score(seq_all):
    if len(seq_all) < 2:
        return 0.0

    motion_values = []

    for t in range(1, len(seq_all)):
        prev = seq_all[t - 1]
        curr = seq_all[t]

        valid = (prev[:, 2] > 0) & (curr[:, 2] > 0)

        if valid.sum() == 0:
            continue

        diff = curr[valid, :2] - prev[valid, :2]
        dist = np.linalg.norm(diff, axis=1)

        motion_values.append(dist.mean())

    if len(motion_values) == 0:
        return 0.0

    return float(np.mean(motion_values))


# ============================================================
# 10. 10초 chunk 판별 함수
# 원본 판단 구조 유지
# ============================================================

def predict_from_chunk(chunk_buffer):
    if len(chunk_buffer) < SEQ_LEN:
        return 0.0, 0.0, 0, 0, 0.0, False

    seq_all = np.array(chunk_buffer, dtype=np.float32)

    motion_score = compute_motion_score(seq_all)

    probs_list = []

    for start in range(0, len(seq_all) - SEQ_LEN + 1, WINDOW_STRIDE):
        end = start + SEQ_LEN

        window = seq_all[start:end]
        x = window.reshape(SEQ_LEN, -1)

        x_tensor = torch.tensor(
            x,
            dtype=torch.float32
        ).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = gru_model(x_tensor)
            probs = torch.softmax(logits, dim=1)
            fall_prob = probs[0, 1].item()

        probs_list.append(fall_prob)

    if len(probs_list) == 0:
        return 0.0, 0.0, 0, 0, motion_score, False

    probs_arr = np.array(probs_list, dtype=np.float32)

    avg_prob = float(probs_arr.mean())
    max_prob = float(probs_arr.max())

    fall_windows = int((probs_arr >= FALL_TH).sum())
    total_windows = len(probs_arr)

    gru_says_fall = False

    if fall_windows >= MIN_FALL_WINDOWS and avg_prob >= AVG_FALL_TH:
        gru_says_fall = True

    final_is_fall = False

    if gru_says_fall and motion_score >= MOTION_TH:
        final_is_fall = True

    return avg_prob, max_prob, fall_windows, total_windows, motion_score, final_is_fall


# ============================================================
# 11. 텔레그램 전송 함수
# 메시지는 시간만 전송
# ============================================================

def telegram_ready():
    if not SEND_TELEGRAM:
        return False

    if TELEGRAM_BOT_TOKEN.strip() == "":
        return False

    if TELEGRAM_CHAT_ID.strip() == "":
        return False

    return True


def send_telegram_message(message):
    if not telegram_ready():
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message
    }

    try:
        response = requests.post(url, data=payload, timeout=TELEGRAM_TIMEOUT)

        if response.status_code == 200:
            print("Telegram message sent")
        else:
            print("Telegram message failed:", response.text)

    except Exception as e:
        print("Telegram message error:", e)


def send_telegram_photo(image_path, caption=""):
    if not telegram_ready():
        return

    if not SEND_SNAPSHOT:
        return

    if not os.path.exists(image_path):
        print("Snapshot file does not exist:", image_path)
        return

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
        else:
            print("Telegram photo failed:", response.text)

    except Exception as e:
        print("Telegram photo error:", e)


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


# ============================================================
# 12. 신고/알림 프로토콜
# Telegram 메시지는 시간만 보냄
# ============================================================

def trigger_alert_protocol(snapshot_path=None):
    alert_time_str = now_time_string()

    print("====================================")
    print("FALL ALERT TRIGGERED")
    print("time:", alert_time_str)
    print("====================================")

    message = (
        "🚨 FALL ALERT TRIGGERED\n"
        f"Time: {alert_time_str}"
    )

    send_telegram_message(message)

    if snapshot_path is not None:
        send_telegram_photo(
            snapshot_path,
            caption=f"Fall detected at {alert_time_str}"
        )


# ============================================================
# 13. 화면 canvas 생성
# 글자는 영상 위가 아니라 오른쪽 패널에 표시
# ============================================================

def make_display_canvas(
    display_frame,
    alert_text,
    current_display_text,
    last_result_text,
    last_avg_prob,
    last_max_prob,
    last_fall_windows,
    last_total_windows,
    last_motion_score,
    valid_kpts,
    chunk_len,
    fps,
    alert_triggered,
    alert_elapsed,
    predict_reason
):
    h, w = display_frame.shape[:2]

    canvas = np.zeros((h, w + PANEL_WIDTH, 3), dtype=np.uint8)

    canvas[:, :w] = display_frame

    panel = canvas[:, w:w + PANEL_WIDTH]
    panel[:] = (25, 25, 25)

    if alert_triggered:
        alert_color = (0, 0, 255)
    else:
        alert_color = (0, 255, 0)

    if current_display_text == "FALL":
        result_color = (0, 0, 255)
    elif current_display_text == "NO FALL":
        result_color = (0, 255, 0)
    elif current_display_text == "COLLECTING":
        result_color = (0, 255, 255)
    elif current_display_text == "NO PERSON":
        result_color = (255, 255, 255)
    elif current_display_text == "RESET":
        result_color = (0, 255, 255)
    else:
        result_color = (255, 255, 0)

    x0 = w + 10
    y = 22
    line = 21

    cv2.putText(canvas, "Fall Detection", (x0, y), FONT, 0.50, (255, 255, 255), 1)
    y += line

    cv2.putText(canvas, "GRU Privacy", (x0, y), FONT, 0.42, (200, 200, 200), 1)
    y += line + 2

    cv2.putText(canvas, alert_text, (x0, y), FONT, 0.42, alert_color, 1)
    y += line + 2

    cv2.putText(canvas, f"Current: {current_display_text}", (x0, y), FONT, 0.42, result_color, 1)
    y += line

    cv2.putText(canvas, f"Last: {last_result_text}", (x0, y), FONT, 0.40, result_color, 1)
    y += line

    cv2.putText(canvas, f"avg/max: {last_avg_prob:.2f}/{last_max_prob:.2f}", (x0, y), FONT, 0.38, (255, 255, 255), 1)
    y += line

    cv2.putText(canvas, f"windows: {last_fall_windows}/{last_total_windows}", (x0, y), FONT, 0.38, (255, 255, 255), 1)
    y += line

    cv2.putText(canvas, f"motion: {last_motion_score:.3f}", (x0, y), FONT, 0.38, (255, 255, 255), 1)
    y += line

    cv2.putText(canvas, f"kpts: {valid_kpts}", (x0, y), FONT, 0.38, (255, 255, 255), 1)
    y += line

    cv2.putText(canvas, f"chunk: {chunk_len}/{CHUNK_LEN}", (x0, y), FONT, 0.38, (255, 255, 255), 1)
    y += line

    cv2.putText(canvas, f"reason: {predict_reason}", (x0, y), FONT, 0.34, (255, 255, 255), 1)
    y += line

    cv2.putText(canvas, f"FPS: {fps:.1f}", (x0, y), FONT, 0.38, (255, 255, 255), 1)
    y += line

    cv2.putText(canvas, f"Blur: {SHOW_BLUR}", (x0, y), FONT, 0.38, (255, 255, 255), 1)
    y += line

    cv2.putText(canvas, f"TG: {telegram_ready()}", (x0, y), FONT, 0.38, (255, 255, 255), 1)
    y += line

    cv2.putText(canvas, f"Overlay: {DRAW_YOLO_OVERLAY}", (x0, y), FONT, 0.38, (255, 255, 255), 1)
    y += line

    if alert_triggered and alert_elapsed is not None:
        cv2.putText(canvas, f"Alert: {alert_elapsed:.1f}s", (x0, y), FONT, 0.38, (0, 0, 255), 1)
        y += line

    cv2.line(canvas, (w + 8, y), (w + PANEL_WIDTH - 8, y), (90, 90, 90), 1)
    y += line

    cv2.putText(canvas, "q: quit / r: reset", (x0, y), FONT, 0.35, (220, 220, 220), 1)
    y += line

    cv2.putText(canvas, "Telegram: reset", (x0, y), FONT, 0.35, (220, 220, 220), 1)

    return canvas


# ============================================================
# 14. Picamera2 시작
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

print("Picamera2 시작 완료")


# ============================================================
# 15. Buffer / 상태 변수
# ============================================================

chunk_buffer = deque(maxlen=CHUNK_LEN)

alert_triggered = False
alert_time = None

last_result_text = "NONE"
current_result_text = "COLLECTING"

last_avg_prob = 0.0
last_max_prob = 0.0
last_fall_windows = 0
last_total_windows = 0
last_motion_score = 0.0

no_person_start_time = None

prev_time = time.time()
fps = 0.0

last_update_id = None
last_telegram_check_time = 0.0

last_predict_reason = "NONE"

print("====================================")
print("실시간 낙상 감지 시작 - Privacy Version")
print("기준: 원본 real_time_fall_detection.py 판단 로직 유지")
print(f"SEQ_LEN={SEQ_LEN}, CHUNK_LEN={CHUNK_LEN}, WINDOW_STRIDE={WINDOW_STRIDE}")
print("방식: 10초 수집 -> 20프레임 window 다중 판별 -> buffer 초기화")
print("추가: NO PERSON 발생 시 chunk가 100 미만이어도 20프레임 이상이면 조기 판별")
print("화면: 블러 영상 + 오른쪽 정보 패널")
print("Telegram: 시간만 전송")
print("종료: q 키 / alert reset: r 키")
print("터미널 reset: touch reset_alert.flag")
print("Telegram reset: /reset 또는 reset")
print("====================================")

if telegram_ready():
    send_telegram_message(
        "Fall detection privacy version started\n"
        f"Time: {now_time_string()}"
    )


# ============================================================
# 16. Reset 함수
# ============================================================

def reset_alert_state():
    global alert_triggered
    global alert_time
    global last_result_text
    global current_result_text
    global last_avg_prob
    global last_max_prob
    global last_fall_windows
    global last_total_windows
    global last_motion_score
    global no_person_start_time
    global last_predict_reason

    alert_triggered = False
    alert_time = None
    chunk_buffer.clear()

    no_person_start_time = None

    last_result_text = "RESET"
    current_result_text = "COLLECTING"

    last_avg_prob = 0.0
    last_max_prob = 0.0
    last_fall_windows = 0
    last_total_windows = 0
    last_motion_score = 0.0
    last_predict_reason = "RESET"

    print("ALERT RESET / BUFFER CLEARED")


# ============================================================
# 17. Telegram reset 확인
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
# 18. 판별 실행 공통 함수
# ============================================================

def run_chunk_prediction(predict_reason, display_frame, current_time):
    global alert_triggered
    global alert_time
    global current_result_text
    global last_result_text
    global last_avg_prob
    global last_max_prob
    global last_fall_windows
    global last_total_windows
    global last_motion_score
    global last_predict_reason

    avg_prob, max_prob, fall_windows, total_windows, motion_score, final_is_fall = predict_from_chunk(chunk_buffer)

    last_avg_prob = avg_prob
    last_max_prob = max_prob
    last_fall_windows = fall_windows
    last_total_windows = total_windows
    last_motion_score = motion_score
    last_predict_reason = predict_reason

    if final_is_fall:
        current_result_text = "FALL"
        last_result_text = "FALL"

        if not alert_triggered:
            alert_triggered = True
            alert_time = current_time

            # 원본 저장 금지. 블러된 display_frame만 저장.
            cv2.imwrite(SNAPSHOT_PATH, display_frame)

            trigger_alert_protocol(
                snapshot_path=SNAPSHOT_PATH
            )

    else:
        current_result_text = "NO FALL"
        last_result_text = "NO FALL"

    # 원본 방식 유지:
    # 판별 후 chunk 비움
    chunk_buffer.clear()


# ============================================================
# 19. 실시간 루프
# ============================================================

while True:
    # --------------------------------------------------------
    # 카메라 프레임 획득
    # --------------------------------------------------------
    frame_rgb = picam2.capture_array()

    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    # YOLO 입력은 원본 사용
    yolo_frame = frame_bgr.copy()

    # 사람이 보는 화면은 블러 적용
    if SHOW_BLUR:
        display_frame = apply_privacy_blur(frame_bgr.copy())
    else:
        display_frame = frame_bgr.copy()

    # --------------------------------------------------------
    # YOLO Pose
    # --------------------------------------------------------
    results = pose_model.predict(
        source=yolo_frame,
        imgsz=IMG_SIZE,
        conf=0.25,
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

    # YOLO skeleton은 블러된 화면 위에만 표시
    if DRAW_YOLO_OVERLAY:
        display_frame = draw_skeleton(display_frame, kpts_out)

    current_time = time.time()

    # --------------------------------------------------------
    # 사람이 잡히는 경우: chunk buffer에 skeleton 저장
    # 원본 구조 유지
    # --------------------------------------------------------
    if valid_kpts >= MIN_VALID_KPTS:
        no_person_start_time = None

        kpts_norm = normalize_frame(kpts_out)
        chunk_buffer.append(kpts_norm)

    else:
        if no_person_start_time is None:
            no_person_start_time = current_time

        no_person_elapsed = current_time - no_person_start_time

        # 추가:
        # 사람이 갑자기 안 잡히는 상태가 됐고,
        # 이미 최소 20프레임 이상 모였다면 100프레임을 기다리지 않고 조기 판별
        if (
            len(chunk_buffer) >= SEQ_LEN
            and no_person_elapsed >= NO_PERSON_EVAL_SEC
        ):
            run_chunk_prediction(
                predict_reason="NO_PERSON_EARLY",
                display_frame=display_frame,
                current_time=current_time
            )

            no_person_start_time = current_time

        # 원본의 no person reset 유지
        elif (
            not alert_triggered
            and no_person_elapsed >= NO_PERSON_RESET_SEC
        ):
            chunk_buffer.clear()
            last_result_text = "NO PERSON"
            current_result_text = "NO PERSON"
            last_avg_prob = 0.0
            last_max_prob = 0.0
            last_fall_windows = 0
            last_total_windows = 0
            last_motion_score = 0.0
            last_predict_reason = "NO_PERSON_RESET"

    # --------------------------------------------------------
    # 10초 chunk가 다 찼으면 판별
    # 원본 구조 유지
    # --------------------------------------------------------
    if len(chunk_buffer) >= CHUNK_LEN:
        run_chunk_prediction(
            predict_reason="FULL_CHUNK",
            display_frame=display_frame,
            current_time=current_time
        )

    # --------------------------------------------------------
    # 외부 reset flag 확인
    # 다른 터미널에서 touch reset_alert.flag 입력하면 리셋
    # --------------------------------------------------------
    if os.path.exists(RESET_FLAG_PATH):
        reset_alert_state()
        os.remove(RESET_FLAG_PATH)

        if telegram_ready():
            send_telegram_message(
                "Fall alert reset completed\n"
                f"Time: {now_time_string()}"
            )

    # --------------------------------------------------------
    # Telegram reset 확인
    # 매 프레임마다 하지 않고 5초마다 확인
    # --------------------------------------------------------
    now_for_telegram = time.time()

    if telegram_ready() and (now_for_telegram - last_telegram_check_time >= TELEGRAM_CHECK_INTERVAL):
        last_telegram_check_time = now_for_telegram

        if check_telegram_reset():
            reset_alert_state()

            send_telegram_message(
                "Fall alert reset completed\n"
                f"Time: {now_time_string()}"
            )

    # --------------------------------------------------------
    # 현재 화면 표시 상태
    # --------------------------------------------------------
    if valid_kpts < MIN_VALID_KPTS:
        current_display_text = "NO PERSON"

    elif len(chunk_buffer) < CHUNK_LEN:
        current_display_text = "COLLECTING"

    else:
        current_display_text = current_result_text

    if alert_triggered:
        alert_text = "ALERT: TRIGGERED"
    else:
        alert_text = "ALERT: NOT TRIGGERED"

    # --------------------------------------------------------
    # FPS 계산
    # --------------------------------------------------------
    now = time.time()
    fps = 1.0 / max(now - prev_time, 1e-6)
    prev_time = now

    if alert_triggered and alert_time is not None:
        alert_elapsed = current_time - alert_time
    else:
        alert_elapsed = None

    # --------------------------------------------------------
    # 화면 출력 canvas
    # --------------------------------------------------------
    canvas = make_display_canvas(
        display_frame=display_frame,
        alert_text=alert_text,
        current_display_text=current_display_text,
        last_result_text=last_result_text,
        last_avg_prob=last_avg_prob,
        last_max_prob=last_max_prob,
        last_fall_windows=last_fall_windows,
        last_total_windows=last_total_windows,
        last_motion_score=last_motion_score,
        valid_kpts=valid_kpts,
        chunk_len=len(chunk_buffer),
        fps=fps,
        alert_triggered=alert_triggered,
        alert_elapsed=alert_elapsed,
        predict_reason=last_predict_reason
    )

    cv2.imshow("Fall Detection Privacy", canvas)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    if key == ord("r") or key == ord("R"):
        reset_alert_state()

        if telegram_ready():
            send_telegram_message(
                "Fall alert reset completed\n"
                f"Time: {now_time_string()}"
            )


# ============================================================
# 20. 종료
# ============================================================

picam2.stop()
cv2.destroyAllWindows()

print("종료")
