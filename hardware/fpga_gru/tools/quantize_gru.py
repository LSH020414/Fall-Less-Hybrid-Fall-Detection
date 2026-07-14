#!/usr/bin/env python3
"""Quantize the Raspberry Pi GRU and compare it with a NumPy float model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SEQ_LEN = 20
STRIDE = 5
INPUT_SIZE = 51
HIDDEN_SIZE = 64
INPUT_BITS = 18
INPUT_FRAC = 6
WEIGHT_BITS = 16
WEIGHT_FRAC = 14
HIDDEN_BITS = 16
HIDDEN_FRAC = 14
PRE_FRAC = 12
INPUT_SCALE = 1 << INPUT_FRAC
WEIGHT_SCALE = 1 << WEIGHT_FRAC
HIDDEN_SCALE = 1 << HIDDEN_FRAC
PRE_SCALE = 1 << PRE_FRAC
INPUT_QMIN = -(1 << (INPUT_BITS - 1))
INPUT_QMAX = (1 << (INPUT_BITS - 1)) - 1
WEIGHT_QMIN = -(1 << (WEIGHT_BITS - 1))
WEIGHT_QMAX = (1 << (WEIGHT_BITS - 1)) - 1
HIDDEN_QMIN = -(1 << (HIDDEN_BITS - 1))
HIDDEN_QMAX = (1 << (HIDDEN_BITS - 1)) - 1
SIG_SCALE = 1 << 15
LUT_STEP_SHIFT = 6  # Q?.12 preactivation sampled every 1/64.
LUT_X_MAX_Q = 8 * PRE_SCALE
LUT_SIZE = (2 * LUT_X_MAX_Q >> LUT_STEP_SHIFT) + 1


def quantize_input(values: np.ndarray) -> np.ndarray:
    return np.clip(
        np.rint(values * INPUT_SCALE), INPUT_QMIN, INPUT_QMAX
    ).astype(np.int64)


def quantize_weight(values: np.ndarray) -> np.ndarray:
    return np.clip(
        np.rint(values * WEIGHT_SCALE), WEIGHT_QMIN, WEIGHT_QMAX
    ).astype(np.int64)


def quantize_bias(values: np.ndarray) -> np.ndarray:
    return np.rint(values * PRE_SCALE).astype(np.int64)


def round_shift(values: np.ndarray | np.int64, bits: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64)
    offset = 1 << (bits - 1)
    return np.where(values >= 0, (values + offset) >> bits, -((-values + offset) >> bits))


def sigmoid_float(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def build_luts() -> tuple[np.ndarray, np.ndarray]:
    x_q = np.arange(-LUT_X_MAX_Q, LUT_X_MAX_Q + 1, 1 << LUT_STEP_SHIFT)
    x = x_q.astype(np.float64) / PRE_SCALE
    sig = np.clip(np.rint(sigmoid_float(x) * SIG_SCALE), 0, SIG_SCALE - 1).astype(np.int64)
    tanh = np.clip(
        np.rint(np.tanh(x) * HIDDEN_SCALE), -HIDDEN_SCALE, HIDDEN_SCALE
    ).astype(np.int64)
    assert len(sig) == LUT_SIZE
    return sig, tanh


SIG_LUT, TANH_LUT = build_luts()


def lut_index(x_q: np.ndarray) -> np.ndarray:
    clipped = np.clip(x_q, -LUT_X_MAX_Q, LUT_X_MAX_Q)
    return np.clip(
        (clipped + LUT_X_MAX_Q + (1 << (LUT_STEP_SHIFT - 1))) >> LUT_STEP_SHIFT,
        0,
        LUT_SIZE - 1,
    ).astype(np.int64)


def sigmoid_q(x_q: np.ndarray) -> np.ndarray:
    return SIG_LUT[lut_index(x_q)]


def tanh_q(x_q: np.ndarray) -> np.ndarray:
    return TANH_LUT[lut_index(x_q)]


def load_weights(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key].astype(np.float64) for key in data.files}


def split_gates(array: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return tuple(np.split(array, 3, axis=0))  # PyTorch order: reset, update, new.


class FloatGru:
    def __init__(self, weights: dict[str, np.ndarray]) -> None:
        self.w_ir, self.w_iz, self.w_in = split_gates(weights["gru.weight_ih_l0"])
        self.w_hr, self.w_hz, self.w_hn = split_gates(weights["gru.weight_hh_l0"])
        self.b_ir, self.b_iz, self.b_in = split_gates(weights["gru.bias_ih_l0"])
        self.b_hr, self.b_hz, self.b_hn = split_gates(weights["gru.bias_hh_l0"])
        self.fc_w = weights["fc.weight"]
        self.fc_b = weights["fc.bias"]

    def infer(self, sequence: np.ndarray) -> tuple[float, np.ndarray]:
        h = np.zeros(HIDDEN_SIZE, dtype=np.float64)
        for x in sequence:
            r = sigmoid_float(self.w_ir @ x + self.b_ir + self.w_hr @ h + self.b_hr)
            z = sigmoid_float(self.w_iz @ x + self.b_iz + self.w_hz @ h + self.b_hz)
            n = np.tanh(self.w_in @ x + self.b_in + r * (self.w_hn @ h + self.b_hn))
            h = (1.0 - z) * n + z * h
        logits = self.fc_w @ h + self.fc_b
        probability = float(sigmoid_float(np.array([logits[1] - logits[0]]))[0])
        return probability, logits


class FixedGru:
    def __init__(self, weights: dict[str, np.ndarray]) -> None:
        q = {}
        for name, value in weights.items():
            q[name] = quantize_bias(value) if "bias" in name else quantize_weight(value)
        self.w_ir, self.w_iz, self.w_in = split_gates(q["gru.weight_ih_l0"])
        self.w_hr, self.w_hz, self.w_hn = split_gates(q["gru.weight_hh_l0"])
        self.b_ir, self.b_iz, self.b_in = split_gates(q["gru.bias_ih_l0"])
        self.b_hr, self.b_hz, self.b_hn = split_gates(q["gru.bias_hh_l0"])
        self.fc_w = q["fc.weight"]
        self.fc_b = q["fc.bias"]
        self.quantized = q

    @staticmethod
    def input_linear(w: np.ndarray, x: np.ndarray, bias: np.ndarray) -> np.ndarray:
        product_shift = INPUT_FRAC + WEIGHT_FRAC - PRE_FRAC
        return np.clip(
            round_shift(w @ x, product_shift) + bias,
            -(1 << 30),
            (1 << 30) - 1,
        )

    @staticmethod
    def hidden_linear(w: np.ndarray, h: np.ndarray, bias: np.ndarray) -> np.ndarray:
        product_shift = HIDDEN_FRAC + WEIGHT_FRAC - PRE_FRAC
        return np.clip(
            round_shift(w @ h, product_shift) + bias,
            -(1 << 30),
            (1 << 30) - 1,
        )

    def infer_q(self, sequence_q: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
        h = np.zeros(HIDDEN_SIZE, dtype=np.int64)
        for x in sequence_q:
            r_pre = self.input_linear(self.w_ir, x, self.b_ir) + self.hidden_linear(
                self.w_hr, h, self.b_hr
            )
            z_pre = self.input_linear(self.w_iz, x, self.b_iz) + self.hidden_linear(
                self.w_hz, h, self.b_hz
            )
            r = sigmoid_q(r_pre)
            z = sigmoid_q(z_pre)
            n_input = self.input_linear(self.w_in, x, self.b_in)
            n_hidden = self.hidden_linear(self.w_hn, h, self.b_hn)
            n_pre = n_input + round_shift(r * n_hidden, 15)
            n = tanh_q(n_pre)
            h = round_shift(z * h + (SIG_SCALE - z) * n, 15)
            h = np.clip(h, HIDDEN_QMIN, HIDDEN_QMAX)

        logits = self.hidden_linear(self.fc_w, h, self.fc_b)
        probability_q = int(sigmoid_q(np.array([logits[1] - logits[0]], dtype=np.int64))[0])
        return probability_q, logits, h

    def infer(self, sequence: np.ndarray) -> tuple[float, np.ndarray]:
        probability_q, logits_q, _ = self.infer_q(quantize_input(sequence))
        return probability_q / SIG_SCALE, logits_q.astype(np.float64) / PRE_SCALE


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    frame = frame.astype(np.float64, copy=True)
    xy = frame[:, :2]
    confidence = frame[:, 2:3]
    left_hip, right_hip = frame[11], frame[12]
    if left_hip[2] > 0 and right_hip[2] > 0:
        center = (left_hip[:2] + right_hip[:2]) / 2.0
    else:
        valid = frame[:, 2] > 0
        center = xy[valid].mean(axis=0) if valid.any() else np.zeros(2)

    left_shoulder, right_shoulder = frame[5], frame[6]
    if left_shoulder[2] > 0 and right_shoulder[2] > 0:
        scale = np.linalg.norm(left_shoulder[:2] - right_shoulder[:2])
    elif left_hip[2] > 0 and right_hip[2] > 0:
        scale = np.linalg.norm(left_hip[:2] - right_hip[:2])
    else:
        scale = 1.0
    if scale < 1e-6:
        scale = 1.0

    frame[:, :2] = (xy - center) / scale
    frame[:, 2:3] = confidence
    return frame


def load_records(csv_path: Path) -> list[dict]:
    columns = [
        "video_name",
        "label",
        "label_name",
        "frame_index",
        "joint_id",
        "x",
        "y",
        "confidence",
    ]
    table = pd.read_csv(csv_path, usecols=columns)
    records: list[dict] = []
    for video_name, group in table.groupby("video_name", sort=False):
        frames = []
        for _, frame_group in group.groupby("frame_index", sort=True):
            frame_group = frame_group.sort_values("joint_id")
            frame = np.zeros((17, 3), dtype=np.float64)
            joint_ids = frame_group["joint_id"].to_numpy(dtype=np.int64)
            valid_ids = (joint_ids >= 0) & (joint_ids < 17)
            frame[joint_ids[valid_ids], 0] = frame_group["x"].to_numpy()[valid_ids]
            frame[joint_ids[valid_ids], 1] = frame_group["y"].to_numpy()[valid_ids]
            frame[joint_ids[valid_ids], 2] = frame_group["confidence"].to_numpy()[valid_ids]
            frames.append(normalize_frame(frame).reshape(-1))
        records.append(
            {
                "video_name": video_name,
                "label": int(group["label"].iloc[0]),
                "label_name": str(group["label_name"].iloc[0]),
                "frames": np.asarray(frames, dtype=np.float64),
            }
        )
    return records


def evaluate(records: list[dict], float_model: FloatGru, fixed_model: FixedGru) -> dict:
    comparisons = []
    input_abs_values = []
    for record in records:
        frames = record["frames"]
        input_abs_values.append(np.abs(frames).reshape(-1))
        for start in range(0, len(frames) - SEQ_LEN + 1, STRIDE):
            sequence = frames[start : start + SEQ_LEN]
            float_probability, _ = float_model.infer(sequence)
            fixed_probability, _ = fixed_model.infer(sequence)
            comparisons.append(
                {
                    "video_name": record["video_name"],
                    "start_frame": start,
                    "float_probability": float_probability,
                    "fixed_probability": fixed_probability,
                    "absolute_error": abs(float_probability - fixed_probability),
                    "float_fall_090": float_probability >= 0.90,
                    "fixed_fall_090": fixed_probability >= 0.90,
                }
            )

    errors = np.array([row["absolute_error"] for row in comparisons], dtype=np.float64)
    agreement = np.array(
        [row["float_fall_090"] == row["fixed_fall_090"] for row in comparisons], dtype=bool
    )
    input_values = np.concatenate(input_abs_values)
    return {
        "records": len(records),
        "windows": len(comparisons),
        "mean_probability_error": float(errors.mean()),
        "max_probability_error": float(errors.max()),
        "threshold_090_agreement": float(agreement.mean()),
        "threshold_090_mismatches": int((~agreement).sum()),
        "input_abs_percentiles": {
            "p95": float(np.percentile(input_values, 95)),
            "p99": float(np.percentile(input_values, 99)),
            "p999": float(np.percentile(input_values, 99.9)),
            "max": float(input_values.max()),
        },
        "largest_errors": sorted(comparisons, key=lambda row: row["absolute_error"], reverse=True)[:20],
    }


def write_mem(path: Path, values: np.ndarray, bits: int = 16) -> None:
    mask = (1 << bits) - 1
    width = (bits + 3) // 4
    flat = values.astype(np.int64).reshape(-1)
    path.write_text("\n".join(f"{int(value) & mask:0{width}x}" for value in flat) + "\n")


def export_rtl_assets(out_dir: Path, fixed_model: FixedGru) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, values in fixed_model.quantized.items():
        bits = 32 if "bias" in name else WEIGHT_BITS
        write_mem(out_dir / f"{name.replace('.', '_')}.mem", values, bits=bits)
    write_mem(out_dir / "sigmoid_lut.mem", SIG_LUT)
    write_mem(out_dir / "tanh_lut.mem", TANH_LUT)
    metadata = {
        "input_size": INPUT_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "sequence_length": SEQ_LEN,
        "input_bits": INPUT_BITS,
        "input_fraction_bits": INPUT_FRAC,
        "weight_bits": WEIGHT_BITS,
        "weight_fraction_bits": WEIGHT_FRAC,
        "hidden_bits": HIDDEN_BITS,
        "hidden_fraction_bits": HIDDEN_FRAC,
        "preactivation_fraction_bits": PRE_FRAC,
        "sigmoid_fraction_bits": 15,
        "lut_step_shift": LUT_STEP_SHIFT,
        "lut_size": LUT_SIZE,
        "weight_count": int(sum(value.size for value in fixed_model.quantized.values())),
    }
    (out_dir / "gru_rtl_config.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    weights = load_weights(args.weights)
    float_model = FloatGru(weights)
    fixed_model = FixedGru(weights)
    records = load_records(args.csv)
    result = evaluate(records, float_model, fixed_model)
    args.out.mkdir(parents=True, exist_ok=True)
    export_rtl_assets(args.out / "mem", fixed_model)
    (args.out / "quantization_report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "largest_errors"}, indent=2))


if __name__ == "__main__":
    main()
