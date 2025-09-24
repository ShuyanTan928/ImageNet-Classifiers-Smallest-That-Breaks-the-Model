"""Command-line interface for estimating minimal adversarial perturbations."""
from __future__ import annotations

import argparse
import itertools
import os
import statistics
import time
from collections import defaultdict
from typing import Dict, List

import torch
from tqdm import tqdm

from .eval_min_eps import AttackResult, estimate_min_eps
from .utils import (
    build_transform,
    create_dataloader,
    ensure_dir,
    load_imagenet_dataset,
    load_model,
    save_json,
    set_global_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate minimal adversarial perturbations with PGD")
    parser.add_argument("--dataset-name", default="mrm8488/ImageNet1K-val", help="HuggingFace dataset identifier")
    parser.add_argument("--split", default="train", help="Dataset split to load")
    parser.add_argument("--model-name", default="resnet18", help="Torchvision model name")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of images to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch-size", type=int, default=1, help="Evaluation batch size")
    parser.add_argument("--num-workers", type=int, default=0, help="Dataloader worker count")
    parser.add_argument("--attack-types", nargs="*", default=["untargeted", "targeted"], choices=["untargeted", "targeted"], help="Attack modes to evaluate")
    parser.add_argument("--norms", nargs="*", default=["linf", "l2"], choices=["linf", "l2"], help="Norm constraints to evaluate")
    parser.add_argument("--losses", nargs="*", default=["ce", "cw"], choices=["ce", "cw"], help="Loss functions to evaluate")
    parser.add_argument("--pgd-iters", type=int, default=40, help="Number of PGD iterations per check")
    parser.add_argument("--eps-upper-linf", type=float, default=0.5, help="Upper bound for L_inf search")
    parser.add_argument("--eps-upper-l2", type=float, default=10.0, help="Upper bound for L_2 search")
    parser.add_argument("--binary-search-steps", type=int, default=15, help="Number of binary search iterations")
    parser.add_argument("--output-dir", default="outputs", help="Root directory for experiment outputs")
    parser.add_argument("--device", default=None, help="Device to run on (e.g., cuda, cpu)")
    return parser.parse_args()


def _generate_target(true_label: int, num_classes: int = 1000) -> int:
    candidate = torch.randint(low=0, high=num_classes - 1, size=(1,), dtype=torch.long).item()
    if candidate >= true_label:
        candidate += 1
    return candidate


def _build_run_name(args: argparse.Namespace) -> str:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return f"{args.model_name}_n{args.num_samples}_seed{args.seed}_{timestamp}"


def _summarize_results(records: List[Dict]) -> Dict:
    total = len(records)
    successes = [r for r in records if r["is_success"]]
    success_rate = len(successes) / total if total else 0.0
    eps_values = [r["min_eps"] for r in successes if r["min_eps"] is not None]
    summary = {
        "total_samples": total,
        "successes": len(successes),
        "success_rate": success_rate,
        "min_eps_mean": statistics.mean(eps_values) if eps_values else None,
        "min_eps_median": statistics.median(eps_values) if eps_values else None,
        "min_eps_std": statistics.pstdev(eps_values) if len(eps_values) > 1 else 0.0 if eps_values else None,
    }
    return summary


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_global_seed(args.seed)

    transform = build_transform()
    dataset, selected_indices = load_imagenet_dataset(
        args.dataset_name,
        split=args.split,
        num_samples=args.num_samples,
        seed=args.seed,
        transform=transform,
    )
    dataloader = create_dataloader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    model = load_model(args.model_name, device=device)

    configs = list(itertools.product(args.attack_types, args.losses, args.norms))

    records_by_config: Dict[str, List[Dict]] = defaultdict(list)

    run_name = _build_run_name(args)
    log_dir = os.path.join(args.output_dir, "logs")
    summary_dir = os.path.join(args.output_dir, "summary")
    ensure_dir(log_dir)
    ensure_dir(summary_dir)

    sample_counter = 0
    for batch in tqdm(dataloader, desc="Images", total=len(dataset)):
        images, labels, image_ids = batch
        images = images.to(device)
        labels = labels.to(device)
        batch_size = images.size(0)
        for idx_in_batch in range(batch_size):
            image = images[idx_in_batch : idx_in_batch + 1]
            true_label = labels[idx_in_batch : idx_in_batch + 1]
            image_id = image_ids[idx_in_batch]
            dataset_index = int(selected_indices[sample_counter])
            for attack_type, loss_name, norm in configs:
                if attack_type == "targeted":
                    target_label_value = _generate_target(int(true_label.item()))
                    target_tensor = torch.tensor([target_label_value], device=device)
                else:
                    target_label_value = None
                    target_tensor = None
                eps_upper = args.eps_upper_linf if norm == "linf" else args.eps_upper_l2
                attack_result: AttackResult = estimate_min_eps(
                    model,
                    image,
                    true_label,
                    eps_upper=eps_upper,
                    norm=norm,
                    loss_name=loss_name,
                    pgd_iters=args.pgd_iters,
                    device=device,
                    y_target=target_tensor,
                    binary_search_steps=args.binary_search_steps,
                )
                record = {
                    "image_index": dataset_index,
                    "image_id": image_id,
                    "true_label": int(true_label.item()),
                    "attack_type": attack_type,
                    "loss": loss_name,
                    "norm": norm,
                    "is_success": attack_result.success,
                    "min_eps": attack_result.min_eps,
                    "adv_pred": attack_result.adv_pred,
                    "target_label": target_label_value,
                    "eps_upper": eps_upper,
                    "pgd_iters": args.pgd_iters,
                    "binary_search_steps": args.binary_search_steps,
                    "num_attacks": attack_result.num_attacks,
                }
                key = f"{norm}_{loss_name}_{attack_type}"
                records_by_config[key].append(record)
            sample_counter += 1

    detailed_path = os.path.join(log_dir, f"results_{run_name}.json")
    flat_records = list(itertools.chain.from_iterable(records_by_config.values()))
    metadata = {"args": vars(args), "records": flat_records}
    save_json(detailed_path, metadata)

    for key, records in records_by_config.items():
        summary = _summarize_results(records)
        summary["config"] = key
        summary_path = os.path.join(summary_dir, f"summary_{key}_{run_name}.json")
        save_json(summary_path, {"summary": summary, "records": records})

    print(f"Detailed results saved to {detailed_path}")
    print(f"Summaries saved to {summary_dir}")


if __name__ == "__main__":
    main()
