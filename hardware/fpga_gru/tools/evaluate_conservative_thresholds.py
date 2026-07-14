import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
REPOSITORY = PROJECT.parents[1]
DEFAULT_CSV_PATH = REPOSITORY / "data" / "URFall_all_skeleton_coordinates.csv"
QUANTIZE_PATH = PROJECT / "tools" / "quantize_gru.py"
WEIGHTS_PATH = PROJECT / "model" / "fall2d_gru_weights.npz"
DEFAULT_OUTPUT_PATH = REPOSITORY / "results" / "CONSERVATIVE_PARAMETER_EVALUATION.json"

SEQ_LEN = 20
CHUNK_LEN = 100
STRIDE = 5


def load_quantize_module():
    spec = importlib.util.spec_from_file_location("quantize_gru", QUANTIZE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def motion_score(frames):
    values = []
    for previous, current in zip(frames[:-1], frames[1:]):
        valid = (previous[:, 2] > 0) & (current[:, 2] > 0)
        if not valid.any():
            continue
        distance = np.linalg.norm(
            current[valid, :2] - previous[valid, :2],
            axis=1,
        )
        values.append(float(distance.mean()))
    return float(np.mean(values)) if values else 0.0


def precompute(model, records):
    output = []
    for record in records:
        frames = record["frames"].reshape(-1, 17, 3)
        chunks = []
        for start in range(0, len(frames), CHUNK_LEN):
            chunk = frames[start : start + CHUNK_LEN]
            if len(chunk) < SEQ_LEN:
                continue
            probabilities = []
            for offset in range(0, len(chunk) - SEQ_LEN + 1, STRIDE):
                probability, _ = model.infer(
                    chunk[offset : offset + SEQ_LEN].reshape(SEQ_LEN, -1)
                )
                probabilities.append(float(probability))
            chunks.append(
                {
                    "frames": len(chunk),
                    "probabilities": probabilities,
                    "motion": motion_score(chunk),
                }
            )
        output.append(
            {
                "video_name": record["video_name"],
                "label": int(record["label"]),
                "chunks": chunks,
            }
        )
    return output


def evaluate(records, parameter, full_chunks_only=False):
    predictions = []
    for record in records:
        predicted = False
        for chunk in record["chunks"]:
            if full_chunks_only and chunk["frames"] != CHUNK_LEN:
                continue
            probabilities = chunk["probabilities"]
            high_count = sum(
                value >= parameter["high_threshold"]
                for value in probabilities
            )
            average = float(np.mean(probabilities)) if probabilities else 0.0
            if (
                high_count >= parameter["minimum_high_windows"]
                and average >= parameter["average_threshold"]
                and chunk["motion"] >= parameter["motion_threshold"]
            ):
                predicted = True
                break
        predictions.append(predicted)

    tp = sum(
        row["label"] == 1 and prediction
        for row, prediction in zip(records, predictions)
    )
    tn = sum(
        row["label"] == 0 and not prediction
        for row, prediction in zip(records, predictions)
    )
    fp = sum(
        row["label"] == 0 and prediction
        for row, prediction in zip(records, predictions)
    )
    fn = sum(
        row["label"] == 1 and not prediction
        for row, prediction in zip(records, predictions)
    )
    return {
        **parameter,
        "correct": tp + tn,
        "accuracy": (tp + tn) / len(records),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate conservative fixed-point GRU decision thresholds."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main():
    args = parse_args()
    quantize = load_quantize_module()
    model = quantize.FixedGru(quantize.load_weights(WEIGHTS_PATH))
    records = precompute(model, quantize.load_records(args.csv))

    named_candidates = {
        "current_sensitive": {
            "high_threshold": 0.65,
            "minimum_high_windows": 1,
            "average_threshold": 0.45,
            "motion_threshold": 0.06,
        },
        "balanced_conservative": {
            "high_threshold": 0.80,
            "minimum_high_windows": 3,
            "average_threshold": 0.60,
            "motion_threshold": 0.08,
        },
        "moderate_live": {
            "high_threshold": 0.70,
            "minimum_high_windows": 2,
            "average_threshold": 0.50,
            "motion_threshold": 0.07,
        },
        "strong_conservative": {
            "high_threshold": 0.85,
            "minimum_high_windows": 4,
            "average_threshold": 0.65,
            "motion_threshold": 0.08,
        },
        "original_strict": {
            "high_threshold": 0.90,
            "minimum_high_windows": 5,
            "average_threshold": 0.75,
            "motion_threshold": 0.06,
        },
    }

    all_candidates = []
    for high_threshold in (0.70, 0.75, 0.80, 0.85, 0.90):
        for minimum_high_windows in (2, 3, 4, 5):
            for average_threshold in (0.50, 0.55, 0.60, 0.65, 0.70):
                for motion_threshold in (0.06, 0.08, 0.10, 0.12):
                    parameter = {
                        "high_threshold": high_threshold,
                        "minimum_high_windows": minimum_high_windows,
                        "average_threshold": average_threshold,
                        "motion_threshold": motion_threshold,
                    }
                    all_candidates.append(evaluate(records, parameter))

    all_candidates.sort(
        key=lambda row: (
            row["fp"] == 0,
            row["correct"],
            row["tp"],
            row["minimum_high_windows"],
            row["high_threshold"],
            row["average_threshold"],
            row["motion_threshold"],
        ),
        reverse=True,
    )

    output = {
        "named_candidates": {
            name: {
                "all_record_chunks": evaluate(records, parameter),
                "full_100_frame_chunks_only": evaluate(
                    records,
                    parameter,
                    full_chunks_only=True,
                ),
            }
            for name, parameter in named_candidates.items()
        },
        "best_conservative_candidates": all_candidates[:30],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    for name, result in output["named_candidates"].items():
        print(name)
        print("  all:", result["all_record_chunks"])
        print("  full:", result["full_100_frame_chunks_only"])
    print("TOP")
    for result in all_candidates[:20]:
        print(result)


if __name__ == "__main__":
    main()
