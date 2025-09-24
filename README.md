# Minimum Adversarial Perturbation Estimation (ResNet-18 / ImageNet)

## Overview

This project estimates the smallest perturbation magnitude (epsilon) required to flip a classifier’s decision for a pretrained ResNet-18 (trained on ImageNet). We measure perturbation size under both L_inf and L_2 norms and run attacks using two loss functions: Cross-Entropy (CE) and the Carlini–Wagner (CW) margin. Each image is evaluated with both untargeted and targeted attacks.

Key features:

* Dataset: downloaded automatically from the HuggingFace hub (`mrm8488/ImageNet1K-val`).
* Sampling: configurable number of validation images (default 100) with deterministic seeding.
* Model: pretrained ResNet-18 (ImageNet weights). A local `models/` directory is provided for optional custom checkpoints.
* Attack: PGD (Projected Gradient Descent) with step size alpha = epsilon / 4 and binary search over epsilon.
* Losses: Cross-Entropy (CE) and Carlini–Wagner margin (CW).
* Attack modes: untargeted (flip to any non-ground-truth class) and targeted (assign one random target class per image).

## Repository layout

```
README.md
requirements.txt
models/                # optional directory to store locally downloaded checkpoints
outputs/
  ├─ logs/             # detailed per-sample JSON dumps
  └─ summary/          # aggregate statistics per configuration
scripts/
  ├─ run_attack.py     # main experiment runner
  ├─ utils.py          # dataset/model utilities
  └─ eval_min_eps.py   # binary-search driver around PGD attacks
```

## Dependencies & environment

Create a virtual environment with Python 3.8+ and install the pinned dependencies:

```
pip install -r requirements.txt
```

The requirements include `torch`, `torchvision`, `datasets`, `numpy`, `scipy`, `pillow`, `matplotlib`, and `tqdm`.

## Data

The experiment pulls the ImageNet-1K validation split from HuggingFace using:

```python
from datasets import load_dataset
val_ds = load_dataset("mrm8488/ImageNet1K-val", split="train")
```

Images are transformed to tensors with `Resize(256)`, `CenterCrop(224)`, and `ToTensor()`. You can adjust this behaviour in `scripts/utils.py` if needed.

## High-level experimental flow

1. Randomly sample `--num-samples` images from the ImageNet validation set and record true labels.
2. For each image run both untargeted and targeted attacks. For targeted attacks, sample one random target class different from the true label.
3. For each configuration (norm × loss × attack mode) search for the smallest epsilon that causes the classifier to change its decision. Use binary search backed by PGD feasibility checks.
4. Record per-image results: whether the attack succeeded, minimal epsilon, adversarial prediction, and number of PGD calls.
5. Aggregate results and produce statistics (mean, median, standard deviation, success rate).

## Example usage

Run the full sweep (untargeted/targeted × CE/CW × L_inf/L_2) on 100 validation images:

```
python -m scripts.run_attack \
  --dataset-name mrm8488/ImageNet1K-val \
  --split train \
  --model-name resnet18 \
  --num-samples 100 \
  --pgd-iters 40 \
  --eps-upper-linf 0.5 \
  --eps-upper-l2 10.0 \
  --seed 42
```

Results are stored in `outputs/logs/results_<run_name>.json` and summaries per configuration are written to `outputs/summary/summary_<config>_<run_name>.json`.

## Implementation details & tips

* Input normalization: Use the same mean/std normalization as the model. PGD updates operate in the image space `[0, 1]`.
* Projection onto norm-balls:
  * L_inf: clamp perturbations to `[-epsilon, +epsilon]` elementwise.
  * L_2: rescale perturbations to have norm at most `epsilon`.
* CW loss: implemented as logits margin. Targeted attacks maximise `z_t - max_{i≠t} z_i`, untargeted attacks maximise `max_{i≠y} z_i - z_y`.
* Binary-search edge cases: if the attack fails at the chosen upper bound the sample is marked as failure (minimum epsilon not found).

## Reproducibility

* Fix random seeds for sampling images and target-class selection (`--seed`).
* Record PyTorch/torchvision versions, CUDA and GPU details.
* Save CLI arguments and environment snapshot to `outputs/logs` for each run.

## License

MIT (or supply your organisation’s preferred license).
