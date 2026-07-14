#!/usr/bin/env python3
"""Send one quantized GRU sequence to Basys3 and validate its UART result."""

from __future__ import annotations

import argparse
import struct
import time
from pathlib import Path

import serial


INPUT_COUNT = 20 * 51


def load_case(path: Path, case_index: int) -> list[int]:
    values = [int(line.strip(), 16) for line in path.read_text().splitlines() if line.strip()]
    start = case_index * INPUT_COUNT
    selected = values[start : start + INPUT_COUNT]
    if len(selected) != INPUT_COUNT:
        raise ValueError(f"case {case_index} is incomplete: {len(selected)} values")
    return selected


def build_packet(sequence_id: int, values: list[int]) -> bytes:
    payload = bytearray(struct.pack("<H", sequence_id))
    for raw in values:
        raw &= 0x3FFFF
        payload.extend((raw & 0xFF, (raw >> 8) & 0xFF, (raw >> 16) & 0x03))
    checksum = 0
    for value in payload:
        checksum ^= value
    return bytes((0xA5, 0x5A)) + bytes(payload) + bytes((checksum,))


def read_response(port: serial.Serial, timeout: float) -> bytes:
    deadline = time.monotonic() + timeout
    response = bytearray()
    while time.monotonic() < deadline:
        value = port.read(1)
        if not value:
            continue
        if not response and value[0] != 0xF1:
            continue
        response.extend(value)
        if len(response) == 7:
            return bytes(response)
    raise TimeoutError(f"FPGA response timeout, received={response.hex(' ')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/serial0")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--case", type=int, default=1)
    parser.add_argument("--sequence-id", type=lambda value: int(value, 0), default=0x1234)
    args = parser.parse_args()

    values = load_case(args.vectors, args.case)
    packet = build_packet(args.sequence_id, values)
    with serial.Serial(args.port, args.baud, timeout=0.1) as port:
        port.reset_input_buffer()
        port.write(packet)
        port.flush()
        response = read_response(port, timeout=2.0)

    expected_checksum = 0
    for value in response[:6]:
        expected_checksum ^= value
    if response[6] != expected_checksum:
        raise RuntimeError(f"bad response checksum: {response.hex(' ')}")

    sequence_id, probability = struct.unpack("<HH", response[1:5])
    fall_class = response[5] & 1
    if sequence_id != args.sequence_id:
        raise RuntimeError(f"sequence mismatch: {sequence_id:#06x}")

    print(
        "GRU FPGA UART PASS",
        f"sequence={sequence_id:#06x}",
        f"probability_q15={probability}",
        f"probability={probability / 32768.0:.6f}",
        f"fall={fall_class}",
        f"response={response.hex(' ')}",
    )


if __name__ == "__main__":
    main()
