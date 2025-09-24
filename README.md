# Minimum Adversarial Perturbation Estimation (ResNet-18 / ImageNet)

## Overview

This project estimates the smallest perturbation magnitude (epsilon) required to flip a classifier’s decision for a pretrained ResNet-18 (trained on ImageNet). We measure perturbation size under both L\_inf and L\_2 norms and run attacks using two loss functions: Cross-Entropy (CE) and the Carlini–Wagner (CW) margin. For each image we perform both untargeted and targeted attacks.

Key design choices:

* Dataset: ImageNet-1K validation set — randomly sample 100 images for evaluation.
* Model: pretrained ResNet-18 (ImageNet weights).
* Attack: PGD (Projected Gradient Descent) with step size set to alpha = epsilon / 4.
* Losses: Cross-Entropy (CE) and Carlini–Wagner margin (CW).
* Attack modes: untargeted (flip to any non-ground-truth class) and targeted (each image assigned one random target class among the 999 non-truth classes).

## Recommended repository layout

```
README.md
scripts/
  ├─ run_attack.py         # main experiment runner
  ├─ utils.py              # utilities: model/data loaders, normalization, projections
  └─ eval_min_eps.py       # binary-search / search routine to estimate min eps
outputs/
  ├─ logs/
  └─ figures/
requirements.txt
```

## Dependencies & Environment

Use Python 3.8+ and pin versions for reproducibility. Required packages (examples):

* torch
* torchvision
* numpy
* tqdm
* pillow
* scipy
* matplotlib

List exact versions in `requirements.txt`.

## Data

1. Prepare ImageNet-1K validation images (ILSVRC2012 validation).
2. Provide path via command-line `--val-dir` or edit `utils.py` default.

Note: Only 100 random validation images are used for this study. Fix the random seed to ensure reproducibility.

## High-level experimental flow

1. Randomly sample 100 images from the ImageNet validation set and record true labels.
2. For each image run both untargeted and targeted attacks. For targeted attacks, sample one random target class different from the true label.
3. For each configuration (norm × loss × attack mode) search for the smallest epsilon that causes the classifier to change its decision. Use a search strategy such as binary search with an upper bound and tolerance.

   * Start with a conservative upper bound (e.g., for L\_inf use 0.5 if images are normalized to \[0,1]).
   * Run PGD (step size = epsilon/4, iterations e.g. 40 or 100) to determine whether an attack succeeds at that epsilon.
   * Use binary search to refine epsilon until the desired precision (e.g., 1e-4 for L\_inf, 1e-2 for L\_2).
4. Record per-image results: whether attack succeeded, minimal epsilon, adversarial prediction, number of PGD iterations used.
5. Aggregate results and produce statistics and plots (mean, median, std, success rate, CDFs, boxplots).

## Suggested hyperparameters

* number of samples: 100
* PGD step size: alpha = epsilon / 4
* PGD iterations: 40 (optionally 100 for stricter results)
* binary search max iterations: 15–20
* L\_inf upper bound: 0.5 (normalized \[0,1]); adjust if using \[0,255]
* L\_2 upper bound: choose based on image dimensionality (e.g., test with 100–1000) and refine via quick pilot runs
* random seed: 42

## Evaluation metrics

For each (norm, loss, attack mode) run report:

* number of successful attacks and success rate
* distribution (mean, std, median) of minimal epsilon for successful attacks
* cautionary statistics when including failed attacks (treated as +inf or upper bound)
* visualizations: CDFs and boxplots of minimal epsilon

## Example usage

Command-line example (pseudo):

```
python scripts/run_attack.py \
  --val-dir /path/to/imagenet/val \
  --model resnet18 \
  --norm linf \
  --loss CE \
  --attack_type targeted \
  --num-samples 100 \
  --pgd-iters 40 \
  --eps-upper 0.5 \
  --seed 42
```

`eval_min_eps.py` should implement a per-image routine that returns:
`(is_success, min_eps, adv_pred, num_iters_used)` where `is_success` indicates whether an attack succeeded within the provided upper bound.

## Implementation details & tips

* Input normalization: Use the same mean/std normalization as the model. Make sure the PGD updates and projections operate in the same space used for model input.
* Projection onto norm-balls:

  * L\_inf: clamp perturbation to `[-epsilon, +epsilon]` elementwise.
  * L\_2: if perturbation norm exceeds epsilon, scale it down by `epsilon / ||delta||_2`.
* CW loss: For targeted attacks a typical CW margin is `f(x') = max( max_{i != t} z_i(x') - z_t(x'), -kappa )`. Adjust sign/direction for targeted vs untargeted variants.
* Binary-search edge cases: If attack fails at the chosen upper bound, mark the sample as failure and report that the upper bound was insufficient.

## Output format

Save per-sample JSON entries, e.g.:

```
{
  "image_id": "ILSVRC2012_val_00000001.JPEG",
  "true_label": 281,
  "attack_type": "targeted",
  "loss": "CW",
  "norm": "linf",
  "is_success": true,
  "min_eps": 0.0234,
  "adv_pred": 17
}
```

Also produce aggregated JSON `summary_{norm}_{loss}_{attack}.json` with mean, median, success rate and plot data.

## Reproducibility

* Fix random seeds for sampling images and target-class selection.
* Record PyTorch/torchvision versions, CUDA and GPU details.
* Save CLI arguments and environment snapshot to `outputs/logs` for each run.

## Extensions (optional)

* Scale the experiment to more samples or the entire validation set for stronger statistics.
* Compare models (ResNet-50, ViT, etc.).
* Try different step-size or scheduling strategies for PGD.
* Compare other attacks such as DeepFool, optimization-based CW, or decision-based attacks.

## License & Contact

Apply an appropriate license (e.g., MIT) or your organization’s preferred license.

If you want, I can also generate a starter `scripts/run_attack.py` or `scripts/eval_min_eps.py` implementing the binary search + PGD flow. Tell me which file you want first.

