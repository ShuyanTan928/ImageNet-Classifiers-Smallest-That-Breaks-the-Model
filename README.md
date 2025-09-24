Goal: Estimate the smallest perturbation ϵ (under ℓ∞ and ℓ2 norms) required to flip classification for a
 pretrained model.
 Model: ResNet-18 trained on ImageNet.
 Losses: Cross-Entropy (CE), Carlini–Wagner margin (CW).
 Attack: Sample 100 random images from the ImageNet-1K validation set. Perform both untargeted and
 targeted attacks. For targeted attacks, randomly sample one target class per image from the 999 non-true
 classes. Use PGD (with ϵ/4 as the step size) to implement the attacks
