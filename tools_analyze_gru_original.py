import importlib.util
import json
from pathlib import Path

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[1]
CSV_PATH = WORKSPACE / "URFall_all_skeleton_coordinates.csv"
GRU_PROJECT = Path(r"C:\Users\YangGeon\GRU_Fall_Detect")
QUANTIZE_PATH = GRU_PROJECT / "tools" / "quantize_gru.py"
WEIGHTS_PATH = GRU_PROJECT / "model" / "fall2d_gru_weights.npz"
OUTPUT_PATH = Path(__file__).with_name("GRU_ORIGINAL_MODEL_ANALYSIS.json")

SEQ_LEN = 20
CHUNK_LEN = 100
STRIDE = 5
FALL_TH = 0.90
MIN_FALL_WINDOWS = 5
AVG_FALL_TH = 0.75
MOTION_TH = 0.06


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


def evaluate_chunk(model, flat_frames):
    frames = flat_frames.reshape(-1, 17, 3)
    probabilities = []
    for start in range(0, len(frames) - SEQ_LEN + 1, STRIDE):
        probability, _ = model.infer(
            frames[start : start + SEQ_LEN].reshape(SEQ_LEN, -1)
        )
        probabilities.append(float(probability))

    high_count = sum(value >= FALL_TH for value in probabilities)
    average = float(np.mean(probabilities)) if probabilities else 0.0
    maximum = float(np.max(probabilities)) if probabilities else 0.0
    motion = motion_score(frames)
    prediction = (
        high_count >= MIN_FALL_WINDOWS
        and average >= AVG_FALL_TH
        and motion >= MOTION_TH
    )
    return {
        "frames": len(frames),
        "windows": len(probabilities),
        "average_probability": average,
        "maximum_probability": maximum,
        "high_probability_windows": int(high_count),
        "motion_score": motion,
        "prediction": bool(prediction),
        "_probabilities": probabilities,
    }


def evaluate_record(model, record):
    frames = record["frames"].reshape(-1, 17, 3)
    chunks = []
    for start in range(0, len(frames), CHUNK_LEN):
        chunk = frames[start : start + CHUNK_LEN]
        if len(chunk) < SEQ_LEN:
            continue
        result = evaluate_chunk(model, chunk)
        result["start_frame"] = start
        chunks.append(result)

    prediction = any(chunk["prediction"] for chunk in chunks)
    return {
        "video_name": record["video_name"],
        "label": int(record["label"]),
        "label_name": record["label_name"],
        "prediction": int(prediction),
        "chunks": chunks,
    }


def main():
    quantize = load_quantize_module()
    weights = quantize.load_weights(WEIGHTS_PATH)
    model = quantize.FixedGru(weights)
    records = quantize.load_records(CSV_PATH)

    results = [evaluate_record(model, record) for record in records]
    tp = sum(row["label"] == 1 and row["prediction"] == 1 for row in results)
    tn = sum(row["label"] == 0 and row["prediction"] == 0 for row in results)
    fp = sum(row["label"] == 0 and row["prediction"] == 1 for row in results)
    fn = sum(row["label"] == 1 and row["prediction"] == 0 for row in results)

    missed = []
    for row in results:
        if row["label"] != 1 or row["prediction"] != 0:
            continue
        best = max(
            row["chunks"],
            key=lambda chunk: (
                chunk["high_probability_windows"],
                chunk["average_probability"],
                chunk["maximum_probability"],
            ),
            default={},
        )
        missed.append({"video_name": row["video_name"], "best_chunk": best})

    candidates = []
    for high_threshold in np.arange(0.50, 0.951, 0.05):
        for minimum_high_windows in range(1, 6):
            for average_threshold in np.arange(0.40, 0.801, 0.05):
                for motion_threshold in (0.00, 0.02, 0.04, 0.06):
                    predictions = []
                    for row in results:
                        record_fall = False
                        for chunk in row["chunks"]:
                            probabilities = chunk["_probabilities"]
                            high_count = sum(
                                value >= high_threshold
                                for value in probabilities
                            )
                            if (
                                high_count >= minimum_high_windows
                                and chunk["average_probability"]
                                >= average_threshold
                                and chunk["motion_score"] >= motion_threshold
                            ):
                                record_fall = True
                                break
                        predictions.append(record_fall)

                    candidate_tp = sum(
                        row["label"] == 1 and prediction
                        for row, prediction in zip(results, predictions)
                    )
                    candidate_tn = sum(
                        row["label"] == 0 and not prediction
                        for row, prediction in zip(results, predictions)
                    )
                    candidate_fp = sum(
                        row["label"] == 0 and prediction
                        for row, prediction in zip(results, predictions)
                    )
                    candidate_fn = sum(
                        row["label"] == 1 and not prediction
                        for row, prediction in zip(results, predictions)
                    )
                    candidates.append({
                        "correct": candidate_tp + candidate_tn,
                        "tp": candidate_tp,
                        "tn": candidate_tn,
                        "fp": candidate_fp,
                        "fn": candidate_fn,
                        "high_threshold": round(float(high_threshold), 2),
                        "minimum_high_windows": minimum_high_windows,
                        "average_threshold": round(float(average_threshold), 2),
                        "motion_threshold": motion_threshold,
                    })

    candidates.sort(
        key=lambda row: (
            row["correct"],
            -row["fp"],
            row["tp"],
        ),
        reverse=True,
    )
    best_correct = candidates[0]["correct"]
    best_fp = min(
        row["fp"] for row in candidates if row["correct"] == best_correct
    )
    conservative_best = max(
        (
            row for row in candidates
            if row["correct"] == best_correct and row["fp"] == best_fp
        ),
        key=lambda row: (
            row["minimum_high_windows"],
            row["high_threshold"],
            row["average_threshold"],
            row["motion_threshold"],
        ),
    )

    for row in results:
        for chunk in row["chunks"]:
            chunk.pop("_probabilities", None)

    output = {
        "total": len(results),
        "correct": tp + tn,
        "accuracy": (tp + tn) / len(results),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "fall_recall": tp / (tp + fn),
        "specificity": tn / (tn + fp),
        "parameters": {
            "sequence_length": SEQ_LEN,
            "chunk_length": CHUNK_LEN,
            "stride": STRIDE,
            "fall_threshold": FALL_TH,
            "minimum_fall_windows": MIN_FALL_WINDOWS,
            "average_threshold": AVG_FALL_TH,
            "motion_threshold": MOTION_TH,
        },
        "missed_falls": missed,
        "false_alarms": [
            row["video_name"]
            for row in results
            if row["label"] == 0 and row["prediction"] == 1
        ],
        "best_parameter_candidates": candidates[:20],
        "conservative_best": conservative_best,
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in output.items()
                      if key not in {"missed_falls"}}, indent=2))
    print("MISSED_FALLS")
    for row in missed:
        best = row["best_chunk"]
        print(
            row["video_name"],
            f"high={best.get('high_probability_windows', 0)}",
            f"avg={best.get('average_probability', 0.0):.3f}",
            f"max={best.get('maximum_probability', 0.0):.3f}",
            f"motion={best.get('motion_score', 0.0):.3f}",
        )
    print("BEST_PARAMETER_CANDIDATES")
    for candidate in candidates[:10]:
        print(candidate)


if __name__ == "__main__":
    main()
