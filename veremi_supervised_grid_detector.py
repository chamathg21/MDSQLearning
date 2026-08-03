import argparse
import glob
import json
import math
import os
from collections import defaultdict

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

# Issue 1 / Issue 6 documentation:
# This file provides the missing supervised grid-based classifier described in
# Section V of the paper. It constructs the n x n grid, extracts the 7 grid
# features plus average speed, and evaluates all five VeReMi attack families:
# A1 (constant), A2 (constant offset), A4 (random), A8 (random offset), A16
# (eventual stop).

ATTACK_LABELS = {
    0: "benign",
    1: "constant",
    2: "constant_offset",
    4: "random",
    8: "random_offset",
    16: "eventual_stop",
}


def load_ground_truth(gt_path):
    """Load ground truth labels from a VeReMi ground truth JSON file."""
    labels = {}
    with open(gt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "sender" in record and "attackerType" in record:
                labels[int(record["sender"])] = int(record["attackerType"])
    return labels


def load_received_messages(folder_path):
    """Load all received BSM lines from JSONlog files in a folder."""
    messages = []
    seen = set()
    pattern = os.path.join(folder_path, "JSONlog-*.json")
    for json_path in sorted(glob.glob(pattern)):
        with open(json_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("type") != 3:
                    continue
                sender = int(obj.get("sender", -1))
                message_id = int(obj.get("messageID", -1))
                send_time = float(obj.get("sendTime", obj.get("rcvTime", 0.0)))
                key = (sender, message_id, round(send_time, 6))
                if key in seen:
                    continue
                seen.add(key)
                pos = obj.get("pos", [0.0, 0.0, 0.0])
                spd = obj.get("spd", [0.0, 0.0, 0.0])
                if len(pos) < 2:
                    continue
                messages.append(
                    {
                        "sender": sender,
                        "messageID": message_id,
                        "sendTime": send_time,
                        "pos": (float(pos[0]), float(pos[1])),
                        "speed": (float(spd[0]), float(spd[1])),
                    }
                )
    messages.sort(key=lambda item: item["sendTime"])
    return messages


def build_sender_tracks(messages):
    tracks = defaultdict(list)
    for message in messages:
        tracks[message["sender"]].append(message)
    for sender in tracks:
        tracks[sender].sort(key=lambda item: item["sendTime"])
    return tracks


def find_global_bounds(messages):
    xs = [message["pos"][0] for message in messages]
    ys = [message["pos"][1] for message in messages]
    return min(xs), max(xs), min(ys), max(ys)


def compute_grid_matrix(positions, x_min, x_max, y_min, y_max, grid_size):
    if not positions:
        return np.zeros((grid_size, grid_size), dtype=float)
    dx = (x_max - x_min) / grid_size if grid_size > 0 else 1.0
    dy = (y_max - y_min) / grid_size if grid_size > 0 else 1.0
    grid = np.zeros((grid_size, grid_size), dtype=float)
    for x, y in positions:
        if x < x_min or x > x_max or y < y_min or y > y_max:
            continue
        row = int((y - y_min) // dy)
        col = int((x - x_min) // dx)
        row = min(grid_size - 1, max(0, row))
        col = min(grid_size - 1, max(0, col))
        grid[row, col] += 1.0
    return grid


def average_distance(positions):
    if len(positions) < 2:
        return 0.0
    distances = []
    for i in range(len(positions) - 1):
        dx = positions[i + 1][0] - positions[i][0]
        dy = positions[i + 1][1] - positions[i][1]
        distances.append(math.hypot(dx, dy))
    return sum(distances) / len(distances)


def extract_grid_features_for_sender(positions, speeds, grid_size=60, attack_matrix=None, x_bounds=None, y_bounds=None):
    # Section V features:
    # p_k: total number of positions
    # w_k: number of occupied windows
    # sr_k: spread ratio = w_k / p_k
    # q_k: total points in attack windows
    # ar_k: attack ratio = q_k / p_k
    # avg consecutive distance
    # avg speed
    p_k = float(len(positions))
    avg_consecutive = average_distance(positions)
    avg_speed = float(np.mean([math.hypot(vx, vy) for vx, vy in speeds])) if speeds else 0.0

    if x_bounds is None or y_bounds is None:
        x_values = [pos[0] for pos in positions]
        y_values = [pos[1] for pos in positions]
        x_min, x_max = min(x_values), max(x_values)
        y_min, y_max = min(y_values), max(y_values)
    else:
        x_min, x_max = x_bounds
        y_min, y_max = y_bounds

    grid = compute_grid_matrix(positions, x_min, x_max, y_min, y_max, grid_size)
    w_k = float(np.count_nonzero(grid))
    sr_k = w_k / p_k if p_k > 0 else 0.0
    q_k = float(np.sum(grid * attack_matrix)) if attack_matrix is not None else 0.0
    ar_k = q_k / p_k if p_k > 0 else 0.0

    return np.array([p_k, w_k, sr_k, q_k, ar_k, avg_consecutive, avg_speed], dtype=float)


def build_supervised_dataset(dataset_root, selected_attack_types=None, grid_size=60):
    """Create the per-sender supervised dataset for all five attack types."""
    selected_attack_types = selected_attack_types or [1, 2, 4, 8, 16]
    scenarios = []
    for attack_type in selected_attack_types:
        alias = f"A{attack_type}"
        for folder in sorted(glob.glob(os.path.join(dataset_root, f"*{alias}"))):
            if os.path.isdir(folder):
                scenarios.append(folder)
    scenarios = sorted(dict.fromkeys(scenarios))
    if not scenarios:
        raise FileNotFoundError("No attack scenarios were found for the requested attack types.")

    all_messages = []
    samples = []
    for folder in scenarios:
        gt_path = os.path.join(folder, "GroundTruthJSONlog.json")
        labels = load_ground_truth(gt_path)
        msgs = load_received_messages(folder)
        all_messages.extend(msgs)
        tracks = build_sender_tracks(msgs)

        for sender, track in tracks.items():
            positions = [msg["pos"] for msg in track]
            speeds = [msg["speed"] for msg in track]
            samples.append(
                {
                    "folder": folder,
                    "sender": sender,
                    "label": int(labels.get(sender, 0)),
                    "positions": positions,
                    "speeds": speeds,
                }
            )

    if not all_messages:
        raise ValueError("No VeReMi BSM messages were loaded from the requested scenarios.")

    x_min, x_max, y_min, y_max = find_global_bounds(all_messages)

    # Build a global attack matrix from benign vehicles observed in the training
    # scenarios. This keeps the feature extraction aligned with the paper's grid
    # construction rules.
    benign_presence = np.zeros((grid_size, grid_size), dtype=bool)
    for sample in samples:
        if sample["label"] == 0:
            grid = compute_grid_matrix(sample["positions"], x_min, x_max, y_min, y_max, grid_size)
            benign_presence |= grid > 0
    attack_matrix = (~benign_presence).astype(float)

    X = []
    y = []
    sample_keys = []
    for sample in samples:
        X.append(
            extract_grid_features_for_sender(
                positions=sample["positions"],
                speeds=sample["speeds"],
                grid_size=grid_size,
                attack_matrix=attack_matrix,
                x_bounds=(x_min, x_max),
                y_bounds=(y_min, y_max),
            )
        )
        y.append(sample["label"])
        sample_keys.append((sample["folder"], sample["sender"]))

    X = np.vstack(X)
    y = np.asarray(y)

    train_indices, test_indices = train_test_split(
        np.arange(len(sample_keys)),
        test_size=0.30,
        random_state=42,
        stratify=y,
    )

    return X[train_indices], y[train_indices], X[test_indices], y[test_indices]


def train_and_evaluate(dataset_root, grid_size=60, selected_attack_types=None):
    X_train, y_train, X_test, y_test = build_supervised_dataset(
        dataset_root,
        selected_attack_types=selected_attack_types,
        grid_size=grid_size,
    )

    classifier = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    classifier.fit(X_train, y_train)
    predicted = classifier.predict(X_test)
    macro_f1 = f1_score(y_test, predicted, average="macro")
    report = classification_report(
        y_test,
        predicted,
        target_names=[ATTACK_LABELS.get(label, str(label)) for label in sorted(set(y_test))],
    )
    return macro_f1, report


def main():
    parser = argparse.ArgumentParser(description="Supervised grid-based VeReMi detector for all five attack types.")
    parser.add_argument("--dataset-root", default="Datasets", help="Root directory containing the VeReMi scenario folders.")
    parser.add_argument("--grid-size", type=int, default=60, help="Grid resolution n for the supervised detector.")
    args = parser.parse_args()

    macro_f1, report = train_and_evaluate(
        args.dataset_root,
        grid_size=args.grid_size,
        selected_attack_types=[1, 2, 4, 8, 16],
    )
    print(f"Supervised grid classifier macro F1: {macro_f1:.4f}")
    print(report)


if __name__ == "__main__":
    main()
