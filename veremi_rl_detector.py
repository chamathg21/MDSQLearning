import argparse
import glob
import json
import math
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


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
    patterns = [os.path.join(folder_path, "JSONlog-*.json")]
    for pattern in patterns:
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
                    if len(pos) < 2:
                        continue
                    messages.append(
                        {
                            "sender": sender,
                            "messageID": message_id,
                            "sendTime": send_time,
                            "pos": (float(pos[0]), float(pos[1])),
                        }
                    )
    messages.sort(key=lambda item: item["sendTime"])
    return messages


def kmeans_1d(values, max_iter=20):
    """Simple 1D k-means clustering into 2 clusters."""
    if len(values) < 2:
        return [0] * len(values), (min(values, default=0.0), max(values, default=0.0))
    low = min(values)
    high = max(values)
    if low == high:
        return [0] * len(values), (low, high)

    c1, c2 = low, high
    labels = [0] * len(values)
    for _ in range(max_iter):
        sums = [0.0, 0.0]
        counts = [0, 0]
        changed = False
        for index, value in enumerate(values):
            label = 0 if abs(value - c1) <= abs(value - c2) else 1
            labels[index] = label
            sums[label] += value
            counts[label] += 1
        new_c1 = c1
        new_c2 = c2
        if counts[0] > 0:
            new_c1 = sums[0] / counts[0]
        if counts[1] > 0:
            new_c2 = sums[1] / counts[1]
        if abs(new_c1 - c1) < 1e-6 and abs(new_c2 - c2) < 1e-6:
            break
        c1, c2 = new_c1, new_c2
    return labels, (c1, c2)


def average_distance(positions):
    """Compute average Euclidean displacement between consecutive reported positions."""
    if len(positions) < 2:
        return 0.0
    distances = []
    for i in range(len(positions) - 1):
        dx = positions[i + 1][0] - positions[i][0]
        dy = positions[i + 1][1] - positions[i][1]
        distances.append(math.hypot(dx, dy))
    return sum(distances) / len(distances)


def compute_detection_trajectory(messages, labels, interval=1, alpha=0.5, gamma=0.9, threshold=0.0):
    """Run the online RL-based misbehavior detector and sample accuracy over time."""
    sender_history = defaultdict(list)
    q_tables = defaultdict(
        lambda: {
            ("low", "stay"): 0.0,
            ("low", "switch"): 0.0,
            ("high", "stay"): 0.0,
            ("high", "switch"): 0.0,
        }
    )
    sender_state = {}

    rates = []
    accuracies = []
    last_accuracy = 0.0

    for idx, message in enumerate(messages, start=1):
        sender = message["sender"]
        sender_history[sender].append(message["pos"])

        distances = {}
        for observed_sender, track in sender_history.items():
            if len(track) >= 2:
                distances[observed_sender] = average_distance(track)

        if len(distances) >= 2:
            senders = list(distances.keys())
            values = [distances[s] for s in senders]
            labels_list, centers = kmeans_1d(values)
            if len(set(labels_list)) == 1:
                cluster_state = {senders[i]: "low" for i in range(len(senders))}
            else:
                low_cluster, high_cluster = (0, 1) if centers[0] <= centers[1] else (1, 0)
                cluster_state = {}
                for i, sender_id in enumerate(senders):
                    cluster_state[sender_id] = "low" if labels_list[i] == low_cluster else "high"

            for sender_id, state in cluster_state.items():
                if sender_id in sender_state:
                    prev_state = sender_state[sender_id]
                    action = "stay" if prev_state == state else "switch"
                    if state == "low" and action == "stay":
                        reward = 1.0
                    elif state == "high" and action == "stay":
                        reward = 0.0
                    else:
                        reward = -1.0
                    q_old = q_tables[sender_id][(prev_state, action)]
                    q_next = max(
                        q_tables[sender_id][(state, "stay")],
                        q_tables[sender_id][(state, "switch")],
                    )
                    q_tables[sender_id][(prev_state, action)] = q_old + alpha * (
                        reward + gamma * q_next - q_old
                    )
                sender_state[sender_id] = state

        if idx % interval == 0 or idx == len(messages):
            correct = 0
            total = 0
            for n in range(idx):
                message_sender = messages[n]["sender"]
                if message_sender not in sender_history or len(sender_history[message_sender]) < 2:
                    continue
                state = sender_state.get(message_sender, None)
                q = q_tables[message_sender]
                score = q[("low", "stay")] - q[("high", "stay")] - q[("high", "switch")]
                predicted_malicious = False
                if state == "high":
                    predicted_malicious = True
                elif state == "low":
                    predicted_malicious = False
                else:
                    predicted_malicious = score <= threshold
                true_malicious = labels.get(message_sender, 0) != 0
                if predicted_malicious == true_malicious:
                    correct += 1
                total += 1
            accuracy = correct / total if total > 0 else last_accuracy
            rates.append(idx)
            accuracies.append(accuracy)
            last_accuracy = accuracy

    return rates, accuracies


def plot_accuracy_curve(counts, accuracies, scenario_name, output_path):
    """Save a detection accuracy plot for the selected scenario."""
    plt.figure(figsize=(10, 6))
    plt.plot(counts, accuracies, marker="*", markersize=4, color="green", linewidth=1.0)
    plt.title(f"Detection Accuracy for {scenario_name}")
    plt.xlabel("Number of Basic Safety Messages")
    plt.ylabel("Detection Accuracy")
    plt.ylim(0.0, 1.0)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def scenario_folder_name(scenario):
    return scenario.strip()


def main():
    parser = argparse.ArgumentParser(description="VeReMi RL-based misbehavior detection for random and random offset attacks.")
    parser.add_argument(
        "--scenario",
        required=True,
        help="Name of the folder inside the Datasets directory, e.g. Med1A4, Med1A8, Small1A4, Small1A8",
    )
    parser.add_argument(
        "--dataset-root",
        default="Datasets",
        help="Root folder containing the VeReMi dataset scenario subfolders.",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory where output plots will be written.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=1,
        help="Interval of messages between sampled accuracy points (set to 1 for every message).",
    )
    args = parser.parse_args()

    scenario_name = scenario_folder_name(args.scenario)
    folder_path = os.path.join(args.dataset_root, scenario_name)
    gt_path = os.path.join(folder_path, "GroundTruthJSONlog.json")

    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Scenario folder not found: {folder_path}")
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")

    print(f"Loading dataset from {folder_path}")
    messages = load_received_messages(folder_path)
    labels = load_ground_truth(gt_path)

    print(f"Loaded {len(messages)} unique received BSMs from {scenario_name}")
    print(f"Loaded {len(labels)} ground truth sender labels")

    counts, accuracies = compute_detection_trajectory(
        messages,
        labels,
        interval=args.interval,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{scenario_name}_accuracy.png")
    plot_accuracy_curve(counts, accuracies, scenario_name, output_path)

    if accuracies:
        print(f"Saved detection plot: {output_path}")
        print(f"Final sampled accuracy: {accuracies[-1]:.4f} at {counts[-1]} messages")
    else:
        print("Not enough data to compute detection accuracy.")


if __name__ == "__main__":
    main()
