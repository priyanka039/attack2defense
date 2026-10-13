"""Command line for training, evaluation, and the zero-shot benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from attack2defense import __version__
from attack2defense.config import (
    MODEL_SPECS,
    ROBUST_ACCURACY_TARGET,
    STRONG_EPSILON,
    TARGET_EPSILON,
)
from attack2defense.evaluate import export_results, flag_below_threshold, run_evaluation
from attack2defense.plotting import plot_epsilon_sweep, plot_overall
from attack2defense.train import OPTIMIZERS, SCHEDULERS, TRAINING_MODES, TrainConfig, run_training
from attack2defense.utils import set_progress_enabled


def parse_epsilon(value: str) -> float:
    """Accept ``0.00784`` or a fraction such as ``2/255``."""

    text = str(value).strip()
    try:
        if "/" in text:
            numerator, denominator = text.split("/", 1)
            denominator_value = float(denominator)
            if denominator_value == 0:
                raise ValueError
            return float(numerator) / denominator_value
        return float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"epsilon must be a number or a fraction like 2/255, not {value!r}"
        ) from exc


def _add_shared(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", default="data", help="Where CIFAR-10 is downloaded.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0, help="Use 0 on Windows.")
    parser.add_argument("--quiet", action="store_true", help="Hide per-batch progress bars.")


def _cmd_train(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = "outputs/clean" if args.mode == "clean" else "outputs/defense"
    config = TrainConfig(
        model=args.model,
        mode=args.mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        optimizer_name=args.optimizer,
        scheduler_name=args.scheduler,
        epsilon=args.epsilon,
        pgd_steps=args.pgd_steps,
        adv_ratio=args.adv_ratio,
        val_pgd_steps=args.val_pgd_steps,
        final_pgd_steps=args.final_pgd_steps,
        seed=args.seed,
        num_workers=args.num_workers,
        output_dir=output_dir,
        data_dir=args.data_dir,
        val_per_class=args.val_per_class,
        max_train_batches=args.max_train_batches,
        final_eval=not args.skip_final_eval,
        resume=args.resume,
        num_threads=args.num_threads,
        amp=not args.no_amp,
    )
    run_training(config)


def _cmd_evaluate(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.checkpoint).parent / "test_eval"
    primary = run_evaluation(
        args.checkpoint,
        output_dir,
        epsilon=args.epsilon,
        pgd_steps=args.pgd_steps,
        batch_size=args.batch_size,
        max_per_class=args.max_per_class,
        data_dir=args.data_dir,
        seed=args.seed,
        num_workers=args.num_workers,
        save_examples=not args.no_examples,
        threshold=args.threshold,
    )
    if not args.sweep:
        return

    rows = [primary["summary"]]
    primary_epsilon = float(primary["summary"]["epsilon"])
    for epsilon in (TARGET_EPSILON, 4 / 255, STRONG_EPSILON):
        if abs(epsilon - primary_epsilon) < 1e-8:
            continue
        extra_dir = output_dir / f"eps_{epsilon * 255:.0f}_over_255"
        extra = run_evaluation(
            args.checkpoint,
            extra_dir,
            epsilon=epsilon,
            pgd_steps=args.pgd_steps,
            batch_size=args.batch_size,
            max_per_class=args.sweep_per_class,
            data_dir=args.data_dir,
            seed=args.seed,
            num_workers=args.num_workers,
            save_examples=False,
            threshold=args.threshold,
        )
        rows.append(extra["summary"])

    rows = flag_below_threshold(sorted(rows, key=lambda row: row["epsilon"]), args.threshold)
    sweep_path = export_results(rows, output_dir / "sweep.csv")
    export_results(rows, output_dir / "sweep.json")
    for row in rows:
        if row["pgd_below_threshold"]:
            print(
                f"Below threshold: PGD accuracy {row['pgd_accuracy']:.2%} at epsilon "
                f"{row['epsilon'] * 255:.0f}/255 is under {args.threshold:.0%}.",
                flush=True,
            )
    sweep = pd.DataFrame(rows)
    plot_epsilon_sweep(
        sweep,
        output_dir / "sweep.png",
        "Accuracy as the attack budget grows",
    )
    print(f"Wrote {sweep_path}", flush=True)


def _cmd_compare(args: argparse.Namespace) -> None:
    frames = [pd.read_csv(path) for path in args.summaries]
    combined = pd.concat(frames, ignore_index=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output.with_suffix(".csv"), index=False)
    plot_overall(combined, output, "Clean training vs adversarial training")
    print(combined.to_string(index=False), flush=True)
    print(f"Wrote {output}", flush=True)


def _cmd_benchmark(args: argparse.Namespace) -> None:
    from attack2defense.zeroshot import run_zeroshot_benchmark

    run_zeroshot_benchmark(
        models=args.models,
        output_dir=args.output_dir,
        num_images=args.num_images,
        epsilon=args.epsilon,
        pgd_steps=args.pgd_steps,
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        num_workers=args.num_workers,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="attack2defense",
        description=(
            "Train a CIFAR-10 classifier on all 10 classes and measure accuracy "
            "after FGSM and PGD attacks. Adversarial training is the defense "
            "aimed at staying above 80% for a 2/255 attack."
        ),
    )
    parser.add_argument("--version", action="version", version=f"attack2defense {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser(
        "train",
        help="Train on all 10 CIFAR-10 classes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_shared(train)
    train.add_argument("--model", choices=sorted(MODEL_SPECS), default="small_cnn")
    train.add_argument(
        "--mode",
        choices=TRAINING_MODES,
        default="fgsm",
        help=(
            "clean = ordinary training, fgsm = fast adversarial training, "
            "pgd = multi-step adversarial training."
        ),
    )
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--lr", type=float, default=1e-3, help="AdamW default. Use 0.1 with --optimizer sgd.")
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--optimizer", choices=OPTIMIZERS, default="adamw")
    train.add_argument(
        "--scheduler",
        choices=SCHEDULERS,
        default="cosine",
        help="cosine steps once per epoch; onecycle steps every batch (the GPU recipe).",
    )
    train.add_argument("--no-amp", action="store_true", help="Disable mixed precision on GPU.")
    train.add_argument("--epsilon", type=parse_epsilon, default=TARGET_EPSILON)
    train.add_argument("--pgd-steps", type=int, default=5, help="Steps per batch when --mode pgd.")
    train.add_argument(
        "--adv-ratio",
        type=float,
        default=1.0,
        help="Share of each batch replaced by adversarial images (0 to 1).",
    )
    train.add_argument("--val-pgd-steps", type=int, default=3)
    train.add_argument("--final-pgd-steps", type=int, default=10)
    train.add_argument("--val-per-class", type=int, default=100)
    train.add_argument("--output-dir", default=None)
    train.add_argument("--max-train-batches", type=int, default=None, help="Debug cap on batches per epoch.")
    train.add_argument("--skip-final-eval", action="store_true")
    train.add_argument("--resume", default=None, help="Path to last.pt from an interrupted run.")
    train.add_argument("--num-threads", type=int, default=None)
    train.set_defaults(func=_cmd_train)

    evaluate = subparsers.add_parser(
        "evaluate",
        help="Test a checkpoint on all CIFAR-10 classes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_shared(evaluate)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--output-dir", default=None)
    evaluate.add_argument(
        "--epsilon",
        type=parse_epsilon,
        default=None,
        help="Defaults to the epsilon stored in the checkpoint.",
    )
    evaluate.add_argument("--pgd-steps", type=int, default=10)
    evaluate.add_argument(
        "--threshold",
        type=float,
        default=ROBUST_ACCURACY_TARGET,
        help="Minimum acceptable robust accuracy. Lower results are flagged.",
    )
    evaluate.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Limit the test set. Omit it to use all 10,000 images.",
    )
    evaluate.add_argument("--sweep", action="store_true", help="Also evaluate 2/255, 4/255, and 8/255.")
    evaluate.add_argument("--sweep-per-class", type=int, default=100)
    evaluate.add_argument("--no-examples", action="store_true")
    evaluate.set_defaults(func=_cmd_evaluate)

    compare = subparsers.add_parser(
        "compare",
        help="Plot two or more summary.csv files on one chart.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    compare.add_argument("summaries", nargs="+", help="summary.csv files from evaluate or test_eval.")
    compare.add_argument("--output", default="outputs/comparison.png")
    compare.set_defaults(func=_cmd_compare)

    benchmark = subparsers.add_parser(
        "benchmark",
        help="ImageNet ResNet-50 vs ViT-B/16 on CIFAR cat images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_shared(benchmark)
    benchmark.add_argument(
        "--models",
        nargs="+",
        choices=("resnet50", "vit_b_16"),
        default=["resnet50", "vit_b_16"],
    )
    benchmark.add_argument("--num-images", type=int, default=100)
    benchmark.add_argument("--epsilon", type=parse_epsilon, default=TARGET_EPSILON)
    benchmark.add_argument("--pgd-steps", type=int, default=10)
    benchmark.add_argument("--output-dir", default="outputs/zeroshot")
    benchmark.set_defaults(func=_cmd_benchmark)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    set_progress_enabled(not getattr(args, "quiet", False))
    args.func(args)
