import ast
from collections import deque
from pathlib import Path

import numpy as np


SOURCE_PATH = Path(__file__).with_name("real_time_fall_detection.py")


class FakeUart:
    def __init__(self):
        self.calls = []

    def submit_gru_sequence(self, sequence_id, sequence):
        self.calls.append((sequence_id, np.asarray(sequence)))
        return True


source = SOURCE_PATH.read_text(encoding="utf-8")
tree = ast.parse(source)
wanted_constants = {
    "GRU_SEQ_LEN",
    "GRU_HISTORY_LEN",
    "GRU_SEND_STRIDE",
    "GRU_WINDOWS_PER_CHUNK",
}
nodes = []
for node in tree.body:
    if isinstance(node, ast.Assign):
        names = [
            target.id
            for target in node.targets
            if isinstance(target, ast.Name)
        ]
        if any(name in wanted_constants for name in names):
            nodes.append(node)
    elif isinstance(node, ast.FunctionDef) and node.name == "append_gru_frame":
        nodes.append(node)

namespace = {
    "np": np,
    "deque": deque,
    "uart_ok": True,
    "uart": FakeUart(),
    "gru_buffer": deque(maxlen=100),
    "gru_valid_frame_count": 0,
    "gru_windows_submitted": 0,
    "gru_sequence_id": 0,
}
exec(
    compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE_PATH), "exec"),
    namespace,
)

frame = np.zeros((17, 3), dtype=np.float32)
for _ in range(100):
    namespace["append_gru_frame"](frame)

assert len(namespace["uart"].calls) == 17
assert namespace["gru_valid_frame_count"] == 0
assert namespace["gru_windows_submitted"] == 0
assert len(namespace["gru_buffer"]) == 0
assert namespace["gru_sequence_id"] == 17

for _, sequence in namespace["uart"].calls:
    assert sequence.shape == (20, 17, 3)

print("PI_GRU_CHUNK_BUFFER_TEST PASSED windows=17 buffer=0/100")
