# ImageNet Classifiers: Smallest ε That Breaks the Model

This repository contains a reproducible pipeline for estimating the minimum adversarial perturbation (ε*) required to fool an ImageNet classifier under different attack settings. The project automates sampling images, running Projected Gradient Descent (PGD) based attacks, logging per-example results, and generating plots/tables summarizing success rates for both untargeted and targeted attacks.

## Project structure

```
.
├── model/                  # Pre-trained checkpoints (e.g., adversarially-trained ResNet-18)
├── outputs/
│   ├── examples/           # Original/adversarial image pairs + metadata
│   ├── figures/            # Saved figures for reports
│   ├── logs/               # Detailed JSON logs per experiment
│   ├── plots/              # Auto-generated success-rate plots
│   └── summary/            # Aggregated statistics per configuration
├── scripts/
│   ├── run_attack.py       # End-to-end experiment driver
│   ├── eval_min_eps.py     # Binary search routine for ε*
│   └── utils.py            # Dataset/model helpers
└── requirements.txt        # Python dependencies
```

## Installation

1. **Create a Python environment** (Python 3.9+ recommended).
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Log in to Hugging Face (if needed):** the default dataset loader pulls from `mrm8488/ImageNet1K-val`. If your environment requires authentication, run `huggingface-cli login` before executing experiments.

## Running experiments

The main entry point is `scripts/run_attack.py`, which orchestrates dataset sampling, attack execution, result logging, and visualization.

### Basic usage

```bash
python -m scripts.run_attack \
  --dataset-name mrm8488/ImageNet1K-val \
  --split train \
  --model-name resnet18 \
  --num-samples 100 \
  --seed 42 \
  --attack-types untargeted targeted \
  --norms linf l2 \
  --losses ce cw \
  --pgd-iters 40 \
  --binary-search-steps 15 \
  --output-dir outputs
```

Key arguments:

- `--dataset-name` / `--split`: Hugging Face dataset identifier and split.
- `--num-samples`: Number of images to evaluate (defaults to 100); samples are drawn deterministically given `--seed`.
- `--attack-types`: Which attack modes to evaluate (`untargeted`, `targeted`).
- `--norms`: Perturbation constraints (`linf`, `l2`).
- `--losses`: Optimization losses (`ce` for cross-entropy, `cw` for Carlini–Wagner margin).
- `--pgd-iters`: PGD iterations per feasibility check.
- `--eps-upper-linf` / `--eps-upper-l2`: Upper bounds for the binary search.
- `--binary-search-steps`: Number of refinement steps when solving for ε*.
- `--device`: Override device selection (defaults to CUDA when available).

All outputs are timestamped and written under `--output-dir` with subdirectories for logs, summaries, plots, and example images.

### Outputs

Each run produces:

- **Logs:** `outputs/logs/results_<run_name>.json` contains every per-example attack attempt and metadata for reproducibility.
- **Summaries:** `outputs/summary/summary_<config>_<run_name>.json` stores aggregated statistics, including success rates and ε* distribution moments.
- **Plots:** `outputs/plots/*success_rate.png` visualize success curves for each norm/attack/loss combination.
- **Examples:** The first successful attack saves original/adversarial image pairs plus a text file detailing configuration and ε*.
- **Console tables:** Final terminal output includes median ε* tables for untargeted and targeted settings.

## Reproducing report deliverables

1. Run the default command above (adjust `--num-samples` or search space bounds as needed).
2. Collect the success-rate plots from `outputs/plots/` for the required figures.
3. Use `outputs/summary/` JSON files to fill in tables summarizing success rates and ε* statistics.
4. Reference `outputs/examples/` for qualitative visual comparisons and attack metadata.

## Troubleshooting

- Ensure GPU drivers and CUDA are configured if running on GPU; otherwise set `--device cpu`.
- Large PGD iteration counts or sample sizes can significantly increase runtime—start with smaller values when testing.
- If Hugging Face dataset downloads fail, verify network access and authentication or mirror the dataset locally.

## License

This project is provided for educational purposes in adversarial robustness coursework. Consult upstream datasets/models for their respective licenses.
