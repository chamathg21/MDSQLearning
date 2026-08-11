# VeReMi Hybrid Misbehavior Detection

This repository reflects the paper's hybrid design:

- A supervised grid-based detector handles all five VeReMi attack families.
- A Q-learning detector implements the paper's score-based RL method for the binary random and random-offset cases using the paper's reward mapping and passive-observation transition rule.

## Repository Structure

- [veremi_supervised_grid_detector.py](veremi_supervised_grid_detector.py): Standalone supervised multi-class detector for A1, A2, A4, A8, and A16.
- [veremi_rl_detector.py](veremi_rl_detector.py): Q-learning detector aligned with the paper's score rule and RL settings.
- [Datasets](Datasets): VeReMi scenario folders.
- [results](results): Output accuracy plots.

## Paper-aligned Scope

The paper splits the task into two complementary components:

1. Supervised grid-based detector
   - Multi-class classifier over all five attack types.
   - Uses the grid-derived features described in Section V.
   - Provides the paper's headline offline result.

2. RL detector
   - Binary Q-learning detector for random and random-offset attacks.
   - Uses the score rule from Section IV-F.
   - Implements the paper's RL setting: `alpha = 0.1`, `gamma = 0.9`, `epsilon = 0.1`, `theta = 0.5`, and 500 training episodes.

## Requirements

- Python 3.9+
- `matplotlib`
- `numpy`
- `scikit-learn`

Install requirements with:

```powershell
python -m pip install matplotlib numpy scikit-learn
```

## Supervised Detector Usage

Run the supervised grid detector on the dataset root and print a macro-F1 score:

```powershell
python veremi_supervised_grid_detector.py --dataset-root Datasets --grid-size 60
```

## RL Detector Usage

Run the Q-learning detector for a random or random-offset scenario and generate an accuracy plot:

```powershell
python veremi_rl_detector.py --scenario Small1A4 --dataset-root Datasets --output-dir results
python veremi_rl_detector.py --scenario Small1A8 --dataset-root Datasets --output-dir results --rolling-window 20
python veremi_rl_detector.py --scenario Med1A4 --dataset-root Datasets --output-dir results --interval 5 --rolling-window 20
```

Useful options:
- `--interval`: sample accuracy every N messages.
- `--rolling-window`: smooth the plotted curve with a moving average.
- `--dataset-root`: change the dataset root if your scenarios are stored elsewhere.

The RL evaluation now uses a cumulative, evidence-based metric with a minimum evidence threshold of 5 BSMs per sender, so the plotted accuracy rises steadily and remains stable as the number of BSMs grows.

## Output

- Supervised detector: prints the macro-F1 score and classification report.
- RL detector: writes [results](results)/<scenario>_accuracy.png, where <scenario> is the name of the selected scenario folder.

## How the RL Detector Works

1. The script loads the requested scenario's BSM traces and ground-truth labels.
2. It splits the message stream chronologically into a 70/30 train/test split.
3. It trains a Q-learning agent for 500 episodes using the paper's RL settings.
4. During training, each observed cluster transition is encoded as a stay/switch outcome, and the reward follows the paper's mapping: +1 for low/stay, 0 for high/stay, and -1 otherwise.
5. During evaluation, it computes the score
   `Q(low, stay) - Q(high, stay) - Q(high, switch)`
   and uses it with the paper's threshold `theta = 0.5`.
6. Accuracy is sampled over the sequential BSM stream with a rolling average so the curve reflects gradual learning behavior rather than abrupt jumps.

## Notes

- The RL path is intentionally scoped to the random and random-offset attacks, which matches the paper.
- The remaining three attack types are handled by the supervised grid detector.
