#!/usr/bin/env python3
"""Generate directed GRU RTL vectors from the URFall coordinate CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from quantize_gru import (
    FixedGru,
    FloatGru,
    INPUT_BITS,
    SEQ_LEN,
    STRIDE,
    load_records,
    load_weights,
    quantize_input,
    write_mem,
)


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

    candidates = []
    for record in records:
        frames = record["frames"]
        for start in range(0, len(frames) - SEQ_LEN + 1, STRIDE):
            sequence = frames[start : start + SEQ_LEN]
            float_probability, _ = float_model.infer(sequence)
            probability_q, logits_q, hidden_q = fixed_model.infer_q(quantize_input(sequence))
            candidates.append(
                {
                    "video_name": record["video_name"],
                    "start_frame": start,
                    "label": record["label"],
                    "float_probability": float_probability,
                    "probability_q": probability_q,
                    "logit0_q": int(logits_q[0]),
                    "logit1_q": int(logits_q[1]),
                    "hidden_checksum": int(np.sum(hidden_q, dtype=np.int64)),
                    "sequence_q": quantize_input(sequence),
                }
            )

    selectors = [
        ("minimum", lambda row: row["float_probability"]),
        ("maximum", lambda row: -row["float_probability"]),
        ("near_050", lambda row: abs(row["float_probability"] - 0.50)),
        ("near_090_below", lambda row: abs(row["float_probability"] - 0.90)
         if row["float_probability"] < 0.90 else 10.0),
        ("near_090_above", lambda row: abs(row["float_probability"] - 0.90)
         if row["float_probability"] >= 0.90 else 10.0),
    ]

    selected = []
    used = set()
    for name, key in selectors:
        for row in sorted(candidates, key=key):
            identity = (row["video_name"], row["start_frame"])
            if identity not in used:
                used.add(identity)
                chosen = dict(row)
                chosen["case_name"] = name
                selected.append(chosen)
                break

    args.out.mkdir(parents=True, exist_ok=True)
    all_inputs = np.concatenate([row["sequence_q"].reshape(-1) for row in selected])
    write_mem(args.out / "gru_test_inputs.mem", all_inputs, bits=INPUT_BITS)
    write_mem(
        args.out / "gru_test_expected_probability.mem",
        np.asarray([row["probability_q"] for row in selected], dtype=np.int64),
        bits=16,
    )
    write_mem(
        args.out / "gru_test_expected_logits.mem",
        np.asarray(
            [[row["logit0_q"], row["logit1_q"]] for row in selected], dtype=np.int64
        ),
        bits=32,
    )

    metadata = []
    for row in selected:
        metadata.append({key: value for key, value in row.items() if key != "sequence_q"})
    (args.out / "gru_test_vectors.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
