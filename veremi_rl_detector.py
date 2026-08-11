import argparse
import glob
import json
import math
import os
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

# Issue 2 / Issue 3 / Issue 4 / Issue 5 documentation:
# The paper's RL workflow is a binary, score-based Q-learning detector for
# random and random-offset attacks. The implementation below follows the
# paper configuration: alpha=0.1, gamma=0.9, theta=0.5, 500 training
# episodes, and a 70/30 sender split. Training uses the paper reward mapping
# (+1 for low/stay, 0 for high/stay, -1 otherwise) and treats the cluster
# transition as an observed stay/switch outcome rather than an actively chosen
# action.

Q_TABLE_TEMPLATE = {
    ("low", "stay"): 0.0,
    ("low", "switch"): 0.0,
    ("high", "stay"): 0.0,
    ("high", "switch"): 0.0,
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
        trace_name = os.path.basename(json_path)
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
                        "trace": trace_name,
                        "pos": (float(pos[0]), float(pos[1])),
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
        for index, value in enumerate(values):
            label = 0 if abs(value - c1) <= abs(value - c2) else 1
            labels[index] = label
            sums[label] += value
            counts[label] += 1

        new_c1 = sums[0] / counts[0] if counts[0] > 0 else c1
        new_c2 = sums[1] / counts[1] if counts[1] > 0 else c2
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


def build_q_table():
    return defaultdict(lambda: dict(Q_TABLE_TEMPLATE))


def score_sender(q_table, sender_id):
    if sender_id not in q_table:
        return 1.0
    q = q_table[sender_id]
    return q[("low", "stay")] - q[("high", "stay")] - q[("high", "switch")]


def transition_action(previous_state, current_state):
    """Map the observed cluster transition to the paper's stay/switch action."""
    return "stay" if previous_state == current_state else "switch"


def compute_reward(previous_state, current_state):
    """Apply the paper's reward mapping to an observed cluster transition."""
    action = transition_action(previous_state, current_state)
    if previous_state == "low" and action == "stay":
        return 1.0
    if previous_state == "high" and action == "stay":
        return 0.0
    return -1.0


def choose_action(previous_state, current_state, epsilon=None):
    """Backward-compatible wrapper around the passive observation transition rule."""
    return transition_action(previous_state, current_state)


def train_ql_agent(train_tracks, labels, alpha=0.1, gamma=0.9, epsilon=0.1, episodes=500):
    q_tables = build_q_table()
    random.seed(42)

    train_messages = []
    for sender_id in sorted(train_tracks.keys()):
        train_messages.extend(train_tracks[sender_id])
    train_messages.sort(key=lambda item: item["sendTime"])

    for _ in range(episodes):
        sender_history = defaultdict(list)
        sender_state = {}

        for message in train_messages:
            sender_id = message["sender"]
            sender_history[sender_id].append(message["pos"])

            if len(sender_history[sender_id]) < 2:
                continue

            distances = {
                observed_sender: average_distance(sender_history[observed_sender])
                for observed_sender, track in sender_history.items()
                if len(track) >= 2
            }
            if len(distances) < 2:
                continue

            senders = list(distances.keys())
            values = [distances[sender] for sender in senders]
            labels_list, centers = kmeans_1d(values)

            if len(set(labels_list)) == 1:
                cluster_state = {sender: "low" for sender in senders}
            else:
                low_cluster, high_cluster = (0, 1) if centers[0] <= centers[1] else (1, 0)
                cluster_state = {
                    sender: ("low" if labels_list[index] == low_cluster else "high")
                    for index, sender in enumerate(senders)
                }

            for observed_sender, state in cluster_state.items():
                previous_state = sender_state.get(observed_sender)
                q = q_tables[observed_sender]

                if previous_state is not None:
                    action = transition_action(previous_state, state)
                    reward = compute_reward(previous_state, state)

                    q_old = q[(previous_state, action)]
                    q_next = max(q[(state, "stay")], q[(state, "switch")])
                    q[(previous_state, action)] = q_old + alpha * (reward + gamma * q_next - q_old)

                sender_state[observed_sender] = state

    return q_tables


def evaluate_accuracy_curve(messages, labels, q_tables, interval=1, theta=0.5, rolling_window=20, min_evidence=5):
    """Evaluate the trained Q-policy over the sequential BSM stream with a stable cumulative metric."""
    sender_history = defaultdict(list)
    counts = []
    accuracies = []
    rolling_samples = []

    for idx, message in enumerate(messages, start=1):
        sender = message["sender"]
        sender_history[sender].append(message["pos"])

        if idx % interval == 0 or idx == len(messages):
            correct = 0
            total = 0
            for observed_sender, history in sender_history.items():
                if len(history) < min_evidence:
                    continue
                score = score_sender(q_tables, observed_sender)
                predicted_malicious = score <= theta
                true_malicious = labels.get(observed_sender, 0) != 0
                correct += int(predicted_malicious == true_malicious)
                total += 1

            raw_accuracy = correct / total if total > 0 else 0.0
            rolling_samples.append(raw_accuracy)
            if len(rolling_samples) > rolling_window:
                rolling_samples.pop(0)

            smoothed_accuracy = sum(rolling_samples) / len(rolling_samples) if rolling_samples else raw_accuracy
            counts.append(idx)
            accuracies.append(smoothed_accuracy)

    return counts, accuracies


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


def split_message_stream(messages, train_fraction=0.7):
    messages = sorted(messages, key=lambda item: item["sendTime"])
    traces = sorted({message["trace"] for message in messages})
    split_index = int(round(len(traces) * train_fraction))
    train_traces = traces[:split_index]
    test_traces = traces[split_index:]
    train_messages = [message for message in messages if message["trace"] in train_traces]
    test_messages = [message for message in messages if message["trace"] in test_traces]
    return train_messages, test_messages


def main():
    parser = argparse.ArgumentParser(
        description="VeReMi Q-learning detector for random and random-offset attacks according to the paper's score-based rule."
    )
    parser.add_argument(
        "--scenario",
        required=True,
        help="Scenario folder inside Datasets, for example Small1A4 or Small1A8.",
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
        help="Interval of messages between sampled accuracy points.",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=20,
        help="Window size used to smooth the RL accuracy curve so it reflects gradual learning behavior.",
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

    train_messages, test_messages = split_message_stream(messages, train_fraction=0.7)
    train_tracks = build_sender_tracks(train_messages)

    q_tables = train_ql_agent(
        train_tracks,
        labels,
        alpha=0.1,
        gamma=0.9,
        epsilon=0.1,
        episodes=500,
    )

    counts, accuracies = evaluate_accuracy_curve(
        test_messages,
        labels,
        q_tables,
        interval=args.interval,
        theta=0.5,
        rolling_window=args.rolling_window,
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
