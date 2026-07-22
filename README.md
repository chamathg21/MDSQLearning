# VeReMi Reinforcement Learning Misbehavior Detection

This repository contains a Python implementation of a reinforcement learning-based misbehavior detection system for the VeReMi dataset. It detects random and random-offset position falsification attacks using received Basic Safety Messages (BSMs) from VeReMi scenario logs.

## Repository Structure

- `veremi_rl_detector.py`: Main detector script.
- `Datasets/`: Contains VeReMi dataset scenario folders.
- `README.md`: This file.

## Supported Scenarios

The dataset folder contains scenario subfolders such as:

- `Med1A4`: Medium scenario with random attack.
- `Med1A8`: Medium scenario with random offset attack.
- `Small1A4`: Small scenario with random attack.
- `Small1A8`: Small scenario with random offset attack.

The script uses the ground truth file `GroundTruthJSONlog.json` and all `JSONlog-*.json` received message logs in the selected scenario folder.

## Requirements

- Python 3.9+
- `matplotlib`
- `numpy`

Install requirements with:

```powershell
python -m pip install matplotlib numpy
```

## Usage

Run the detector for a specific scenario and generate an accuracy plot. By default the detector samples accuracy at every BSM count (`--interval 1`), so the plot has a point for each message processed.

```powershell
python veremi_rl_detector.py --scenario Med1A4 --dataset-root Datasets --output-dir results
```

Example commands:

```powershell
python veremi_rl_detector.py --scenario Med1A4 --output-dir results
python veremi_rl_detector.py --scenario Med1A8 --output-dir results
python veremi_rl_detector.py --scenario Small1A4 --output-dir results
python veremi_rl_detector.py --scenario Small1A8 --output-dir results
```

If you need fewer points for a very large dataset, use `--interval 5` or higher. Otherwise, keep the default to plot every message.

## Output

The script produces a plot named `results/<scenario>_accuracy.png` showing detection accuracy over the number of BSMs processed.

## How It Works

1. The script loads received BSM messages from `JSONlog-*.json` files in the requested scenario.
2. It loads ground truth attacker labels from `GroundTruthJSONlog.json`.
3. The detector tracks per-sender message position histories.
4. It computes average movement distances for vehicles with at least two reports.
5. It performs a simple 1D k-means clustering on average distances to assign vehicles to `low` or `high` movement clusters.
6. Each sender maintains a Q-table with two states (`low`, `high`) and two actions (`stay`, `switch`).
7. Q-learning updates are applied online as new messages arrive.
8. Accuracy is sampled every N messages and plotted over time.

## Reproducing Results

1. Ensure the `Datasets/` folder is present and includes the chosen scenario.
2. Run the script with the desired `--scenario`.
3. Inspect the generated plot in `results/`.

If you want to tune detection performance, adjust `--interval` or modify the reward and threshold logic in `veremi_rl_detector.py`.
