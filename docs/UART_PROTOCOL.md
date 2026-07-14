# Raspberry Pi ↔ FPGA UART 프로토콜

## 물리 설정

- 115200 baud
- 8N1
- no flow control
- 3.3V logic
- TX, RX, GND만 연결

## Raspberry Pi → FPGA

20 frame × 51 feature = 1020 feature를 전송한다.

| 순서 | 크기 | 내용 |
|---|---:|---|
| SOF0 | 1 byte | `0xA5` |
| SOF1 | 1 byte | `0x5A` |
| Sequence ID | 2 bytes | little endian |
| Feature payload | 3060 bytes | 1020 × signed 18-bit, 각 3 byte |
| Checksum | 1 byte | sequence와 payload의 XOR |

18-bit feature 저장 순서:

```text
byte0 = bit 7:0
byte1 = bit 15:8
byte2 = bit 17:16
```

FPGA는 checksum이 일치하고 GRU core가 idle일 때만 추론을 시작한다.

## FPGA → Raspberry Pi

| 순서 | 크기 | 내용 |
|---|---:|---|
| SOF | 1 byte | `0xF1` |
| Sequence ID | 2 bytes | little endian |
| Probability | 2 bytes | unsigned Q15 |
| Fall class | 1 byte | `0` 또는 `1` |
| Checksum | 1 byte | 앞 6 byte의 XOR |

`probability = probability_q15 / 32768.0`

Fall class는 단일 GRU window의 결과가 아니라 17-window 누적 조건이 완료된 시점의 최종 결과다.

