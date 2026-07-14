# Fall-Less 과제 보고서

## 1. 개발 목적

본 과제의 목적은 고령자의 실내 낙상 사고를 조기에 감지하고 보호자나 관리자에게 자동으로 알림을 보내 사고 대응 시간을 줄이는 것이다.

고령 인구와 독거노인 가구가 증가하면서 낙상은 개인의 건강 문제를 넘어 사회적 돌봄 비용과 연결되는 문제로 커지고 있다. 특히 욕실은 물기, 수증기, 미끄러운 바닥 때문에 사고 가능성이 높지만 사용자가 가장 민감하게 사생활 침해를 느끼는 공간이기도 하다.

일반 홈 카메라는 사람의 모습을 직접 촬영하므로 녹화와 영상 노출에 대한 거부감이 크다. mmWave 또는 4D 레이더는 영상 노출 문제는 작지만, 욕실처럼 반사체가 많고 멀티패스가 발생하는 환경에서는 오탐과 미탐 가능성이 존재한다.

Fall-Less는 이 두 방식의 한계를 줄이기 위해 원본 영상 자체가 아닌 skeleton coordinate를 핵심 정보로 사용한다. 영상에서 관절 좌표를 추출한 뒤, 소프트웨어 TCN과 FPGA GRU가 각각 시간에 따른 자세 변화를 판단한다.

## 2. 개발 목표

1. 원본 영상의 외부 전송과 상시 저장을 최소화한다.
2. skeleton 좌표만으로 낙상 동작의 시간적 특징을 판단한다.
3. Raspberry Pi 소프트웨어와 FPGA 하드웨어가 독립적으로 낙상을 판단한다.
4. 두 판단 결과를 결합해 신뢰도가 높은 긴급 알림을 생성한다.
5. 최종 하드웨어가 SoC 또는 ASIC으로 확장 가능한 구조임을 검증한다.

## 3. 시스템 구성

### 3.1 Raspberry Pi 5

Raspberry Pi는 카메라 입력, YOLO Pose, TCN 추론, FPGA UART 통신, 화면 표시와 알림을 담당한다.

카메라는 320×240, 10fps로 동작한다. YOLO Pose가 17개 관절의 x, y, confidence를 생성하며, 이 좌표는 TCN 입력과 FPGA GRU 입력으로 각각 변환된다.

### 3.2 Basys3 FPGA

Basys3는 UART packet을 수신하고, 20 frame skeleton sequence를 고정소수점 GRU에 입력한다. GRU 출력 확률과 100 frame 구간의 동작량을 누적한 후 최종 낙상 여부를 결정한다.

낙상 확정 시:

- LD0 ON
- 7-segment에 `FALL` 표시
- UART로 Raspberry Pi에 fall class 전송

## 4. 소프트웨어 TCN 알고리즘

TCN은 시간 방향의 합성곱을 사용해 연속적인 자세 변화를 분석한다.

- 입력 길이: 100 frame
- frame당 feature: 51
- feature 구성: 17 joints × `(x, y, confidence)`
- channel: 64 → 128 → 128
- dilation: 1 → 2 → 4
- dropout: 0.2
- 출력: no-fall/fall 2 class

TCN의 fall probability가 0.70 이상인 결과가 2회 연속 발생하면 TCN 낙상으로 확정한다. 한 번의 불안정한 좌표나 순간적인 자세 변화가 즉시 경보로 이어지는 것을 줄이기 위한 조건이다.

## 5. 하드웨어 GRU 알고리즘

GRU는 이전 frame의 hidden state를 유지하면서 현재 skeleton과 과거 skeleton의 관계를 계산한다. FPGA에서는 실수 연산 대신 정수와 고정소수점 연산을 사용한다.

### 5.1 GRU 구조

- input size: 51
- hidden size: 64
- sequence length: 20
- input: signed 18-bit
- weight: signed 16-bit
- hidden state: signed 16-bit
- activation: sigmoid/tanh LUT
- output: fall probability Q15

### 5.2 자원 절약 방식

모든 GRU neuron에 별도의 곱셈기를 배치하지 않고, MAC 연산기를 시간 분할해 반복 사용한다. 이 방식은 계산 시간이 조금 늘어나지만 Basys3의 DSP와 LUT 사용량을 크게 줄인다.

### 5.3 최종 누적 판단

Raspberry Pi는 100 frame에서 길이 20의 window를 stride 5로 생성한다. 총 17개 window가 FPGA로 전달된다.

FPGA는 다음 조건을 모두 만족할 때 낙상을 확정한다.

```text
high_count >= 2
AND probability_average >= 0.50
AND motion_average >= 0.07
```

여기서 high window는 GRU 확률이 0.70 이상인 window다. 한 번의 높은 확률만으로
확정하지 않고 2개 이상의 window에서 반복 확인한다. 이 설정은 단발성
YOLO Pose 오류를 차단하면서 보수형보다 실제 낙상 감지 조건을 낮춘다.

기존 데이터셋 최적화 설정은 `high_count >= 1`, 평균 확률 `0.45`, 동작량
`0.06`, high window 기준 `0.65`였다. 이 설정은 115/118을 기록했지만,
실시간 시연에서는 민감형과 보수형 사이의 중간 설정을 최종 배포 프로필로 선택했다.

## 6. 이중 판단 방식

TCN과 GRU는 같은 skeleton을 사용하지만 서로 다른 구조와 시간 길이로 판단한다.

- TCN: 긴 100 frame 흐름을 소프트웨어에서 분석
- GRU: 짧은 20 frame 변화들을 FPGA에서 반복 분석

한쪽에서만 낙상이 검출되면 해당 모듈의 상태를 표시한다. 두 모듈이 모두 낙상을 확인하면 `CALL POLICE` 긴급 경고를 생성한다. 이는 한 모델의 일시적인 오판이 전체 시스템의 최종 경보로 바로 이어지는 것을 줄이기 위한 구조다.

## 7. 통신

Raspberry Pi와 FPGA는 115200 baud UART로 연결된다.

Raspberry Pi는 20×51개의 고정소수점 feature와 sequence ID, checksum을 전송한다. FPGA는 sequence ID, Q15 확률, 최종 class와 checksum을 응답한다.

통신 packet에 시작 byte와 XOR checksum을 사용해 byte 순서 오류나 불완전 packet을 검출한다.

## 8. 검증

### 8.1 Python 정량 평가

GRU 고정소수점 evaluator를 118개 record에 적용했다. 데이터셋에 직접
최적화한 민감형 프로필의 결과는 다음과 같다.

- correct: 115/118
- accuracy: 97.46%
- TP: 75
- TN: 40
- FP: 0
- FN: 3

현재 실기 중간형 프로필의 동일 데이터셋 참고 결과는 104/118,
`TP/TN/FP/FN = 64/40/0/14`이다. 단발성 오검출 억제와 실시간 낙상
감지력 사이의 절충값이다.

### 8.2 RTL simulation

다음 항목을 Vivado simulation으로 확인했다.

- GRU 고정소수점 기준 벡터 5개 일치
- UART packet parsing과 checksum
- motion tracker의 frame transition 계산
- 17-window 누적 판단
- 평균 확률 및 동작량 미달 시 경보 차단
- top-level UART response

### 8.3 FPGA implementation

- synthesis 완료
- place and route 완료
- 100MHz timing 충족
- WNS +0.127ns
- bitstream 생성
- Basys3 실기 program 및 동작 확인

## 9. 시연

시연에서는 Raspberry Pi 카메라로 낙상 영상을 촬영했다. VNC 화면에서 관절 검출 상태, TCN buffer, GRU history, UART 상태와 각 모델의 결과를 확인했다.

FPGA 낙상 확정 시 Basys3의 LD0와 7-segment가 동작했고, Raspberry Pi가 UART 응답을 수신했다. TCN과 FPGA가 모두 낙상을 확정한 경우 화면에 긴급 경고창이 표시되었다.

## 10. 한계와 향후 개선

1. 115/118 결과는 현재 평가 record에 맞춘 민감형 parameter 결과이며, 현재 배포된 중간형 프로필과 구분해야 한다.
2. 카메라 각도, 사람 가림, YOLO Pose 좌표 손실은 입력 품질에 영향을 준다.
3. Raspberry Pi의 CPU 추론 속도는 모델 크기와 화면 출력에 따라 달라질 수 있다.
4. 실제 제품화 전에는 다양한 욕실 구조, 조명, 체형과 일상동작 데이터로 추가 검증해야 한다.
5. 최종적으로는 FPGA prototype을 SoC/ASIC으로 축소하고 무선 알림 모듈과 결합할 수 있다.

## 11. 결론

본 과제는 영상 기반 pose estimation, TCN 소프트웨어 추론, UART 통신, 고정소수점 GRU RTL과 FPGA 실기 검증을 하나의 시스템으로 연결했다.

원본 영상 대신 skeleton coordinate를 판단 데이터로 사용하고, Raspberry Pi와 FPGA가 독립적으로 낙상을 판정하는 이중 구조를 구현했다는 점에 의의가 있다.
