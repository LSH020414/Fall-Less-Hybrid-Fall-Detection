# Raspberry Pi TCN 소프트웨어

## 역할

Raspberry Pi 소프트웨어는 다음 기능을 하나의 실시간 loop에서 수행한다.

1. Picamera2 영상 수집
2. YOLO Pose 관절 검출
3. skeleton 좌표 정규화
4. 100 frame TCN buffer 관리
5. TCN 낙상 판단
6. 20 frame GRU window 생성 및 UART 전송
7. FPGA 응답 수신
8. VNC 상태 화면 및 로컬 경고 표시

## TCN 입력

한 frame은 17개 COCO keypoint로 구성된다.

```text
[joint0_x, joint0_y, joint0_confidence, ..., joint16_confidence]
```

따라서 입력 tensor는 `batch × 100 × 51`이다. 모델 입력 직전에 PyTorch Conv1d 형식인 `batch × 51 × 100`으로 변환한다.

## TCN 구성

| 항목 | 값 |
|---|---|
| Sequence | 100 frame |
| Input dimension | 51 |
| Channel | 64, 128, 128 |
| Dilation | 1, 2, 4 |
| Dropout | 0.2 |
| Class | no-fall, fall |
| Threshold | 0.70 |
| Confirmation | 2회 연속 |

## 좌표 유효성

- keypoint confidence threshold: 0.30
- 최소 유효 관절: 8개
- 사람이 일정 시간 검출되지 않으면 buffer와 alert 상태를 reset

## 알림 정책

- TCN 단독 검출: TCN 상태 표시
- FPGA 단독 검출: FPGA 상태 표시
- TCN + FPGA 검출: 긴급 로컬 경고
- Telegram과 snapshot은 선택 기능이며 기본값은 OFF

## 실행 코드

[real_time_fall_detection.py](../software/raspberry_pi/real_time_fall_detection.py)

YOLO와 TCN weight는 저장소에 포함하지 않는다. 파일을 다음 위치에 배치하거나 환경변수로 경로를 지정한다.

```text
software/raspberry_pi/models/yolo11n_blur_pose_best.pt
software/raspberry_pi/models/best_tcn_videowise.pt
```

