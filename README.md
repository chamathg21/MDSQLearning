# VeReMi Q-Learning Position Falsification Detector

This project implements a lightweight reinforcement-learning detector for position falsification attacks using the VeReMi dataset layout found under the Datasets folder.

## What it does

- Reads Basic Safety Messages (BSMs) from a selected VeReMi simulation folder.
- Uses the message stream in temporal order.
- Tracks each sender's position history.
- Computes average inter-report distance as a simple motion feature.
- Applies a two-state Q-learning policy to classify senders as benign or malicious.
- Produces a plot of detection accuracy versus the number of processed BSMs.

## Requirements

Install the Python dependencies:

```bash
pip install matplotlib
```

## Running the detector

From the repository root, run:

```bash
python detect_position_falsification_rl.py --dataset Med1A1
```

This uses the simulation folder `Datasets/Med1A1` and writes an image file such as:

```bash
outputs/Med1A1_accuracy.png
```

### Optional arguments

- `--dataset <name>`: choose a simulation folder under `Datasets`
- `--root <path>`: change the datasets root if needed
- `--output <path>`: output directory for the generated graph
- `--evaluation-interval <N>`: evaluate accuracy every `N` BSMs
- `--alpha <value>`: Q-learning learning rate
- `--gamma <value>`: Q-learning discount factor
- `--threshold <value>`: decision threshold for the learned score

### Example with a custom folder

```bash
python detect_position_falsification_rl.py --dataset Small1A1 --output results --evaluation-interval 25
```

## Notes

- The detector uses the ground truth file in each simulation folder to evaluate accuracy.
- The current implementation applies the RL idea in a simplified, lightweight form suitable for experimentation and extension.
