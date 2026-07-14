import ast
import queue
import struct
import threading
import time
import unittest
from pathlib import Path

import numpy as np


SOURCE_PATH = Path(__file__).with_name("real_time_fall_detection.py")


def load_uart_class():
    source = SOURCE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "FPGAUart"
    )
    constant_nodes = []
    wanted = {
        "GRU_SEQ_LEN",
        "GRU_INPUT_DIM",
        "GRU_INPUT_SCALE",
        "GRU_INPUT_MIN",
        "GRU_INPUT_MAX",
    }
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id in wanted:
            constant_nodes.append(node)

    namespace = {
        "np": np,
        "queue": queue,
        "struct": struct,
        "threading": threading,
        "time": time,
    }
    module = ast.Module(body=constant_nodes + [class_node], type_ignores=[])
    exec(compile(module, str(SOURCE_PATH), "exec"), namespace)
    constants = {name: namespace[name] for name in wanted}
    return namespace["FPGAUart"], constants


class FakeSerial:
    def __init__(self, data):
        self.data = bytearray(data)

    def read(self, size):
        if not self.data:
            return b""
        chunk = self.data[:size]
        del self.data[:size]
        return bytes(chunk)


class FPGAUartProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.uart_class, cls.constants = load_uart_class()

    def test_gru_packet_layout_and_checksum(self):
        sequence = np.zeros(
            (
                self.constants["GRU_SEQ_LEN"],
                self.constants["GRU_INPUT_DIM"],
            ),
            dtype=np.float32,
        )
        sequence[0, 0] = -1.5
        packet = self.uart_class.build_gru_packet(0x1234, sequence)

        self.assertEqual(len(packet), 3065)
        self.assertEqual(packet[:4], bytes((0xA5, 0x5A, 0x34, 0x12)))
        self.assertEqual(packet[4:7], bytes((0xA0, 0xFF, 0x03)))
        self.assertEqual(packet[-1], self.uart_class.xor_checksum(packet[2:-1]))

    def test_fpga_response_decode(self):
        response = bytearray((0xF1, 0x34, 0x12, 0xF5, 0x7F, 0x01))
        response.append(self.uart_class.xor_checksum(response))

        uart = self.uart_class.__new__(self.uart_class)
        uart.ser = FakeSerial(response)
        result = uart._read_gru_response(timeout=0.1)

        self.assertEqual(result["sequence_id"], 0x1234)
        self.assertEqual(result["probability_q15"], 0x7FF5)
        self.assertAlmostEqual(result["probability"], 0x7FF5 / 32768.0)
        self.assertTrue(result["fall"])

    def test_bad_checksum_is_rejected(self):
        response = bytes((0xF1, 0x01, 0x00, 0x00, 0x40, 0x00, 0x00))
        uart = self.uart_class.__new__(self.uart_class)
        uart.ser = FakeSerial(response)

        with self.assertRaises(TimeoutError):
            uart._read_gru_response(timeout=0.01)


if __name__ == "__main__":
    unittest.main(verbosity=2)
