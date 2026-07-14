# 검증 결과

## 1. 데이터셋 최적화 프로필

GRU 고정소수점 evaluator에서 118개 record에 직접 맞춘 parameter는 다음과 같다.

```json
{
  "high_probability": 0.65,
  "minimum_high_windows": 1,
  "average_probability": 0.45,
  "motion": 0.06
}
```

Confusion matrix:

| 실제 \ 예측 | Fall | No-fall |
|---|---:|---:|
| Fall | TP 75 | FN 3 |
| No-fall | FP 0 | TN 40 |

- accuracy: 115/118, 97.46%
- fall recall: 96.15%
- no-fall specificity: 100.00%

이 결과는 현재 118개 record에 대해 선택된 parameter 결과다. 새로운 환경의 일반화 정확도를 의미하지는 않는다.

## 2. 실기 중간형 프로필

실시간 카메라에서는 단일 frame의 skeleton 오인식이 큰 좌표 이동으로 나타날
수 있다. 단일 오인식은 차단하면서 감지력을 확보하기 위해 현재 Basys3에는
다음 조건을 적용했다.

```json
{
  "high_probability": 0.70,
  "minimum_high_windows": 2,
  "average_probability": 0.50,
  "motion": 0.07
}
```

동일한 118개 record에 대한 참고 결과는 104/118,
`TP/TN/FP/FN = 64/40/0/14`이다. 한 번의 높은 확률만으로는 경보가
확정되지 않으며, 기존 보수형보다 낙상 감지 조건을 낮췄다.

## 3. RTL simulation

| Testbench | 확인 항목 | 결과 |
|---|---|---|
| `tb_gru_fixed_core.sv` | Python 고정소수점 기준 벡터 5개 | PASS |
| `tb_gru_motion_tracker.sv` | 19개 frame transition 동작량 | PASS |
| `tb_gru_chunk_decision.sv` | 중간형 확률·평균·동작량 및 단발성 경보 차단 | PASS |
| `tb_gru_fall_detector_top.sv` | UART부터 응답까지 top 통합 | PASS |

## 4. FPGA implementation

| Resource | 사용량 | 비율 |
|---|---:|---:|
| LUT | 4,347 / 20,800 | 20.90% |
| FF | 4,210 / 41,600 | 10.12% |
| BRAM | 17.5 / 50 | 35.00% |
| DSP | 7 / 90 | 7.78% |

Timing:

- clock: 100MHz
- WNS: +0.127ns
- failing endpoint: 0
- timing constraints met

DRC error는 0건이다. 잔여 메시지는 DSP pipeline과 clock buffer 연결에 대한
성능 개선 권고이며 bitstream 생성과 100MHz timing 통과에는 영향을 주지 않는다.

## 5. 실기 검증

1. Basys3 JTAG 연결 확인
2. `GRU_Fall_Detect_MODERATE_FINAL.bit` program 성공
3. Raspberry Pi UART packet 전송
4. FPGA 확률 응답 수신
5. 낙상 시 LD0와 `FALL` 표시
6. TCN과 FPGA 동시 검출 시 로컬 긴급 경고
