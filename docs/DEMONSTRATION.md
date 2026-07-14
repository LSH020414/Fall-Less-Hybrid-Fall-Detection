# 시연 시나리오

## 준비

1. Raspberry Pi 카메라 연결
2. Basys3 bitstream program
3. Pi TX → JA1, Pi RX ← JA2, GND 연결
4. Raspberry Pi service 시작
5. RealVNC로 상태 화면 확인

## 정상 상태

- `Current: NO PERSON` 또는 `NO FALL`
- UART 상태 True
- TCN buffer와 GRU history가 증가
- Basys3 LD0 OFF
- 7-segment blank

## 낙상 상태

1. 카메라가 skeleton을 검출한다.
2. TCN probability가 threshold를 넘고 2회 확인된다.
3. FPGA가 17개 GRU window를 누적한다.
4. FPGA 조건이 충족되면 LD0 ON, `FALL` 표시
5. Raspberry Pi가 FPGA fall response를 수신한다.
6. 두 모듈의 판단이 모두 완료되면 `CALL POLICE` 경고창이 표시된다.

## 재시작

1. Raspberry Pi desktop의 Reset 실행
2. Basys3 BTNC를 눌러 FPGA latch 해제
3. 새로운 TCN/GRU buffer 수집 시작

시연 영상: [fall_less_demo.mp4](../assets/fall_less_demo.mp4)

