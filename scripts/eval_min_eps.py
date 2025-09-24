"""Binary search routine for estimating minimum perturbation magnitudes."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .utils import imagenet_normalization


@dataclass
class AttackResult:
    success: bool
    min_eps: Optional[float]
    adv_pred: Optional[int]
    num_attacks: int
    adv_image: Optional[torch.Tensor]


def _normalize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean) / std


def _cw_margin(logits: torch.Tensor, labels: torch.Tensor, targeted: bool) -> torch.Tensor:
    batch_size = logits.size(0)
    one_hot = torch.zeros_like(logits).scatter_(1, labels.view(-1, 1), 1.0)
    if targeted:
        targeted_logits = torch.sum(logits * one_hot, dim=1)
        max_others = torch.max(logits.masked_fill(one_hot.bool(), float("-inf")), dim=1).values
        return targeted_logits - max_others
    original_logits = torch.sum(logits * one_hot, dim=1)
    max_others = torch.max(logits.masked_fill(one_hot.bool(), float("-inf")), dim=1).values
    return max_others - original_logits


def _project(delta: torch.Tensor, eps: float, norm: str) -> torch.Tensor:
    if eps <= 0:
        return torch.zeros_like(delta)
    if norm == "linf":
        return torch.clamp(delta, -eps, eps)
    if norm == "l2":
        flat = delta.view(delta.size(0), -1)
        norms = torch.norm(flat, p=2, dim=1, keepdim=True)
        factor = torch.minimum(torch.ones_like(norms), eps / (norms + 1e-12))
        projected = flat * factor
        return projected.view_as(delta)
    raise ValueError(f"Unsupported norm: {norm}")


def _attack_success(
    model: torch.nn.Module,
    x: torch.Tensor,
    y_true: torch.Tensor,
    y_target: Optional[torch.Tensor],
    eps: float,
    norm: str,
    loss_name: str,
    pgd_iters: int,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> Tuple[bool, int, torch.Tensor]:
    if eps <= 0:
        with torch.no_grad():
            logits = model(_normalize(x, mean, std))
            pred = torch.argmax(logits, dim=1)
        if y_target is None:
            return (pred != y_true).item(), pred.item(), x.clone()
        return (pred == y_target).item(), pred.item(), x.clone()

    alpha = eps / 4.0
    delta = torch.zeros_like(x, requires_grad=True)

    for _ in range(pgd_iters):
        adv = torch.clamp(x + delta, 0.0, 1.0)
        logits = model(_normalize(adv, mean, std))
        
        is_targeted = y_target is not None
        
        if loss_name == "ce":
            if is_targeted:
                loss = F.cross_entropy(logits, y_target)
            else:
                loss = -F.cross_entropy(logits, y_true)
                
        elif loss_name == "cw":
            margin = _cw_margin(logits, y_target if is_targeted else y_true, is_targeted)
            loss = -margin.mean()
            
        else:
            raise ValueError(f"Unknown loss: {loss_name}")
            
        grad = torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False)[0]

        if norm == "linf":
            direction = grad.sign()
        elif norm == "l2":
            flat = grad.view(grad.size(0), -1)
            grad_norm = flat.norm(p=2, dim=1).view(-1, 1, 1, 1)
            direction = grad / (grad_norm + 1e-12)
        else:
            raise ValueError(f"Unsupported norm: {norm}")
            
        step = -alpha * direction
        
        delta = delta + step
        delta = _project(delta.detach(), eps, norm)
        delta.requires_grad_()

    adv = torch.clamp(x + delta, 0.0, 1.0)
    with torch.no_grad():
        logits = model(_normalize(adv, mean, std))
        pred = torch.argmax(logits, dim=1)
        
    if y_target is None:
        return (pred != y_true).item(), pred.item(), adv
    return (pred == y_target).item(), pred.item(), adv


def estimate_min_eps(
    model: torch.nn.Module,
    x: torch.Tensor,
    y_true: torch.Tensor,
    *,
    eps_upper: float,
    norm: str,
    loss_name: str,
    pgd_iters: int,
    device: torch.device,
    y_target: Optional[torch.Tensor] = None,
    binary_search_steps: int = 15,
) -> AttackResult:
    mean, std = imagenet_normalization()
    mean = mean.to(device)
    std = std.to(device)

    attack_calls = 0

    def run_attack(epsilon: float) -> Tuple[bool, int, torch.Tensor]:
        nonlocal attack_calls
        attack_calls += 1
        return _attack_success(
            model, x, y_true, y_target, epsilon, norm, loss_name, pgd_iters, mean, std
        )

    success, adv_pred, adv_image = run_attack(eps_upper)
    if not success:
        return AttackResult(False, None, None, attack_calls, None)

    lo, hi = 0.0, eps_upper
    best_eps = eps_upper
    best_pred = adv_pred
    best_adv_image = adv_image
    
    for _ in range(binary_search_steps):
        mid = (lo + hi) / 2.0
        success, adv_pred, adv_image = run_attack(mid)
        if success:
            best_eps = mid
            best_pred = adv_pred
            best_adv_image = adv_image
            hi = mid
        else:
            lo = mid
            
    return AttackResult(True, best_eps, best_pred, attack_calls, best_adv_image)
