"""CIFAR-10 training, including the adversarial-training defense."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR

from attack2defense.attacks import craft_adversarial_examples
from attack2defense.config import (
    CIFAR10_CLASSES,
    IMAGENETTE_CLASSES,
    MODEL_SPECS,
    ROBUST_ACCURACY_TARGET,
    TARGET_EPSILON,
)
from attack2defense.data import default_image_size, load_dataset
from attack2defense.evaluate import evaluate_loader, run_evaluation
from attack2defense.models import (
    CONV_MODELS,
    build_pixel_model,
    optimizer_groups,
    parameter_count,
    use_channels_last,
)
from attack2defense.plotting import plot_history
from attack2defense.smoothing import add_gaussian_noise
from attack2defense.utils import (
    device_label,
    get_device,
    load_checkpoint,
    make_logger,
    progress,
    set_seed,
)

TRAINING_MODES = ("clean", "fgsm", "pgd", "gaussian")
OPTIMIZERS = ("adamw", "sgd")
SCHEDULERS = ("cosine", "onecycle")


class TrainingCancelled(Exception):
    """Raised inside the training loop when ``TrainingHooks.should_stop`` returns True."""


@dataclass
class TrainingHooks:
    """Callbacks that let a caller such as the API follow and cancel a run."""

    on_batch: Callable[[int, int, int], None] | None = None
    on_epoch: Callable[[dict], None] | None = None
    should_stop: Callable[[], bool] | None = None


@dataclass
class TrainConfig:
    model: str = "small_cnn"
    dataset: str = "cifar10"
    mode: str = "fgsm"
    epochs: int = 20
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    optimizer_name: str = "adamw"
    scheduler_name: str = "cosine"
    epsilon: float = TARGET_EPSILON
    pgd_steps: int = 5
    adv_ratio: float = 1.0
    sigma: float = 0.25
    val_pgd_steps: int = 3
    final_pgd_steps: int = 10
    seed: int = 42
    num_workers: int = 0
    output_dir: str = "outputs/defense"
    data_dir: str = "data"
    val_per_class: int = 100
    max_train_batches: int | None = None
    final_eval: bool = True
    resume: str | None = None
    num_threads: int | None = None
    amp: bool = True


def attack_tag(
    mode: str, epsilon: float, pgd_steps: int = 5, adv_ratio: float = 1.0, sigma: float = 0.25
) -> str:
    """Short label for the attack a model was trained on, e.g. ``pgd5_eps2_ratio1``."""

    if mode == "clean":
        return "clean"
    if mode == "gaussian":
        return f"gaussian_sigma{sigma:g}"
    attack = "fgsm_rs" if mode == "fgsm" else f"pgd{pgd_steps}"
    return f"{attack}_eps{round(epsilon * 255, 2):g}_ratio{adv_ratio:g}"


def build_run_config(config: TrainConfig) -> dict:
    spec = MODEL_SPECS[config.model]
    payload = asdict(config)
    payload.update(
        {
            "tag": attack_tag(config.mode, config.epsilon, config.pgd_steps, config.adv_ratio, config.sigma),
            "dataset": config.dataset,
            "image_size": default_image_size(config.dataset, spec.image_size),
            "mean": list(spec.mean),
            "std": list(spec.std),
            "num_classes": len(CIFAR10_CLASSES if config.dataset == "cifar10" else IMAGENETTE_CLASSES),
            "class_names": list(CIFAR10_CLASSES if config.dataset == "cifar10" else IMAGENETTE_CLASSES),
            "pretrained": spec.pretrained,
            "description": spec.description,
            "robust_accuracy_target": ROBUST_ACCURACY_TARGET,
        }
    )
    return payload


def _validate(config: TrainConfig) -> None:
    if config.model not in MODEL_SPECS:
        known = ", ".join(sorted(MODEL_SPECS))
        raise ValueError(f"Unknown model '{config.model}'. Choose from: {known}")
    if config.dataset not in ("cifar10", "imagenette"):
        raise ValueError("dataset must be cifar10 or imagenette")
    if config.mode not in TRAINING_MODES:
        raise ValueError(f"mode must be one of {TRAINING_MODES}")
    if config.optimizer_name not in OPTIMIZERS:
        raise ValueError(f"optimizer must be one of {OPTIMIZERS}")
    if config.scheduler_name not in SCHEDULERS:
        raise ValueError(f"scheduler must be one of {SCHEDULERS}")
    if config.epochs < 1:
        raise ValueError("epochs must be at least 1")
    if not 0 <= config.epsilon <= 1:
        raise ValueError("epsilon must be between 0 and 1")
    if config.mode == "pgd" and config.pgd_steps < 1:
        raise ValueError("pgd mode needs pgd_steps >= 1")
    if not 0 <= config.adv_ratio <= 1:
        raise ValueError("adv_ratio must be between 0 and 1")
    if config.mode == "gaussian" and config.sigma <= 0:
        raise ValueError("gaussian mode needs sigma > 0")
    if config.val_per_class < 1:
        raise ValueError("val_per_class must be at least 1")


def _make_optimizer(model: torch.nn.Module, config: TrainConfig) -> torch.optim.Optimizer:
    groups = optimizer_groups(model.backbone, config.model, config.lr, config.weight_decay)
    if config.optimizer_name == "adamw":
        return AdamW(groups)
    return SGD(groups, momentum=0.9, nesterov=True)


def _make_scheduler(optimizer: torch.optim.Optimizer, config: TrainConfig, steps_per_epoch: int):
    """Return the scheduler and whether it steps after every batch."""

    if config.scheduler_name == "onecycle":
        total_steps = config.epochs * steps_per_epoch
        if total_steps < 2:
            raise ValueError("The onecycle scheduler needs at least two training batches.")
        scheduler = OneCycleLR(
            optimizer,
            max_lr=[group["lr"] for group in optimizer.param_groups],
            total_steps=total_steps,
            # OneCycleLR divides by zero unless 1 / total_steps < pct_start < 1.
            pct_start=max(0.25, 1.5 / total_steps),
        )
        return scheduler, True
    return CosineAnnealingLR(optimizer, T_max=config.epochs), False


def _steps_per_epoch(loader, config: TrainConfig) -> int:
    steps = len(loader)
    if config.max_train_batches is not None:
        steps = min(steps, config.max_train_batches)
    return max(steps, 1)


def _training_inputs(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    config: TrainConfig,
) -> torch.Tensor:
    """Clean images for standard training, adversarial images for the defenses.

    With ``adv_ratio`` below 1, only the first share of the (shuffled) batch is
    attacked and the rest stays clean.
    """

    if config.mode == "gaussian":
        return add_gaussian_noise(images, config.sigma)
    if config.mode == "clean" or config.epsilon == 0:
        return images
    count = round(config.adv_ratio * images.shape[0])
    if count == 0:
        return images
    kind = "fgsm_rs" if config.mode == "fgsm" else "pgd"
    attacked = craft_adversarial_examples(
        model, images[:count], labels[:count], config.epsilon, kind=kind, steps=config.pgd_steps
    )
    if count == images.shape[0]:
        return attacked
    return torch.cat([attacked, images[count:]])


def train_one_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    scheduler,
    step_per_batch: bool,
    device: torch.device,
    config: TrainConfig,
    use_amp: bool,
    hooks: TrainingHooks | None = None,
    epoch: int = 1,
) -> tuple[float, float]:
    model.train()
    hooks = hooks or TrainingHooks()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    total_batches = _steps_per_epoch(loader, config)
    autocast = torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext()

    bar = progress(loader, desc="Train")
    for batch_index, (images, labels) in enumerate(bar):
        if config.max_train_batches is not None and batch_index >= config.max_train_batches:
            break
        if hooks.should_stop is not None and hooks.should_stop():
            raise TrainingCancelled
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast:
            inputs = _training_inputs(model, images, labels, config)
            logits = model(inputs)
            loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if step_per_batch:
            scheduler.step()

        batch_size = int(labels.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_correct += int(logits.detach().argmax(dim=1).eq(labels).sum().item())
        total_seen += batch_size
        bar.set_postfix(loss=f"{total_loss / total_seen:.3f}", acc=f"{total_correct / total_seen:.1%}")
        if hooks.on_batch is not None:
            hooks.on_batch(epoch, batch_index + 1, total_batches)

    if total_seen == 0:
        raise RuntimeError("Training saw no images. Check max_train_batches and the data split.")
    return total_loss / total_seen, total_correct / total_seen


def _describe_run(log, config: TrainConfig, device: torch.device, use_amp: bool) -> None:
    spec = MODEL_SPECS[config.model]
    log(f"Device: {device_label(device)}  |  mixed precision: {'on' if use_amp else 'off'}")
    log(spec.description)
    if device.type == "cpu" and config.model == "resnet18":
        log(
            "ResNet-18 adversarial training is about 100 minutes per epoch on a laptop CPU. "
            "Use a GPU, or small_cnn for a CPU run."
        )
    if device.type == "cpu" and config.model in {"resnet50", "vit_b_16"}:
        log("224x224 ResNet-50 or ViT-B/16 training on a CPU is impractically slow. Use a GPU.")
    log(
        f"Training mode: {config.mode}. "
        f"Attack budget epsilon = {config.epsilon:.6f} ({config.epsilon * 255:.2f}/255). "
        f"Target: at least {ROBUST_ACCURACY_TARGET:.0%} accuracy after PGD on the test set."
    )
    log(
        f"Optimizer: {config.optimizer_name}, lr {config.lr:g}, weight decay {config.weight_decay:g}, "
        f"scheduler {config.scheduler_name}, {config.epochs} epochs, batch {config.batch_size}."
    )
    if config.mode == "fgsm":
        log(
            "Defense: fast adversarial training. Each batch is replaced by a one-step "
            "FGSM example that starts from a random point inside the epsilon ball."
        )
    elif config.mode == "pgd":
        log(f"Defense: PGD adversarial training with {config.pgd_steps} steps per batch.")
    elif config.mode == "gaussian":
        log(
            f"Defense: Gaussian noise training (sigma={config.sigma:g}) for randomized smoothing. "
            "This is the certified-defense track, not empirical adversarial training."
        )
    else:
        log("Standard training on clean images. Validation still measures FGSM and PGD.")
    if config.mode != "clean" and config.adv_ratio < 1:
        log(f"Adversarial share of each batch: {config.adv_ratio:.0%}. The rest stays clean.")


def run_training(config: TrainConfig, hooks: TrainingHooks | None = None) -> dict:
    """Train on all 10 CIFAR-10 classes and keep the best validation checkpoint.

    ``hooks`` reports progress per batch and per epoch. If ``should_stop``
    returns True, :class:`TrainingCancelled` propagates after ``last.pt`` of
    the previous epoch is already on disk.
    """

    _validate(config)
    hooks = hooks or TrainingHooks()
    if config.num_threads is not None:
        torch.set_num_threads(config.num_threads)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log = make_logger(output_dir / "train.log")
    set_seed(config.seed)
    device = get_device()
    use_amp = config.amp and device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    run_config = build_run_config(config)
    (output_dir / "config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    _describe_run(log, config, device, use_amp)

    log(f"Loading {config.dataset}. The first run downloads the dataset into the data folder.")
    spec = MODEL_SPECS[config.model]
    image_size = default_image_size(config.dataset, spec.image_size)
    data = load_dataset(
        config.dataset,
        data_dir=config.data_dir,
        image_size=image_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=config.seed,
        val_per_class=config.val_per_class,
        device=device,
    )
    model = build_pixel_model(config.model, num_classes=len(data.class_names), pretrained=None)
    model.to(device)
    if config.model in CONV_MODELS:
        use_channels_last(model)
    log(f"Parameters: {parameter_count(model):,}")

    optimizer = _make_optimizer(model, config)
    scheduler, step_per_batch = _make_scheduler(
        optimizer, config, _steps_per_epoch(data.train_loader, config)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    completed = 0
    best_score = -1.0

    if config.resume:
        resumed = load_checkpoint(config.resume, device)
        if resumed["config"]["model"] != config.model or resumed["config"]["mode"] != config.mode:
            raise ValueError("The resume checkpoint was trained with a different model or mode.")
        if config.scheduler_name == "onecycle" and resumed["config"]["epochs"] != config.epochs:
            raise ValueError(
                "A OneCycle schedule is fixed to its epoch count. Resume with "
                f"--epochs {resumed['config']['epochs']}."
            )
        model.backbone.load_state_dict(resumed["backbone"])
        optimizer.load_state_dict(resumed["optimizer"])
        scheduler.load_state_dict(resumed["scheduler"])
        if "scaler" in resumed:
            scaler.load_state_dict(resumed["scaler"])
        completed = int(resumed["epoch"])
        best_score = float(resumed.get("best_score", -1.0))
        log(f"Resumed from {config.resume} at epoch {completed}.")

    history_path = output_dir / "history.csv"
    if config.resume is None and history_path.exists():
        history_path.unlink()
    epoch_seconds: list[float] = []

    try:
        for epoch_index in range(completed, config.epochs):
            epoch_number = epoch_index + 1
            started = time.perf_counter()

            train_loss, train_accuracy = train_one_epoch(
                model,
                data.train_loader,
                optimizer,
                scaler,
                scheduler,
                step_per_batch,
                device,
                config,
                use_amp,
                hooks=hooks,
                epoch=epoch_number,
            )
            validation = evaluate_loader(
                model,
                data.val_loader,
                epsilon=config.epsilon,
                pgd_steps=config.val_pgd_steps,
                device=device,
                class_names=data.class_names,
                desc=f"Val {epoch_number}",
            )
            current_lr = max(group["lr"] for group in optimizer.param_groups)
            if not step_per_batch:
                scheduler.step()
            elapsed = time.perf_counter() - started
            epoch_seconds.append(elapsed)

            row = {
                "epoch": epoch_number,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_clean": validation["clean_accuracy"],
                "val_fgsm": validation["fgsm_accuracy"],
                "val_pgd": validation["pgd_accuracy"],
                "lr": current_lr,
                "seconds": elapsed,
            }
            pd.DataFrame([row]).to_csv(history_path, mode="a", header=not history_path.exists(), index=False)

            remaining = config.epochs - epoch_number
            eta = (sum(epoch_seconds) / len(epoch_seconds)) * remaining / 60
            log(
                f"Epoch {epoch_number:02d}/{config.epochs}  "
                f"loss {train_loss:.3f}  train {train_accuracy:.2%}  "
                f"val clean {validation['clean_accuracy']:.2%}  "
                f"val FGSM {validation['fgsm_accuracy']:.2%}  "
                f"val PGD {validation['pgd_accuracy']:.2%}  "
                f"lr {current_lr:.2e}  {elapsed:.0f}s" + (f"  ETA {eta:.0f} min" if remaining > 0 else "")
            )

            if (
                config.mode == "fgsm"
                and validation["fgsm_accuracy"] > 0.5
                and validation["fgsm_accuracy"] - validation["pgd_accuracy"] > 0.2
            ):
                log(
                    "Warning: FGSM accuracy is far above PGD accuracy, a sign of catastrophic "
                    "overfitting in one-step training. Retrain with --mode pgd if it persists."
                )

            score = validation["clean_accuracy"] if config.mode == "clean" else validation["pgd_accuracy"]
            is_best = score > best_score
            if is_best:
                best_score = score
            completed = epoch_number
            payload = {
                "backbone": model.backbone.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": completed,
                "best_score": best_score,
                "config": run_config,
                "metrics": row,
            }
            torch.save(payload, output_dir / "last.pt")
            if is_best:
                torch.save(payload, output_dir / "best.pt")
                log(f"Saved new best checkpoint (validation score {best_score:.2%}).")
            if hooks.on_epoch is not None:
                hooks.on_epoch(row)
    except KeyboardInterrupt:
        log("Interrupted. Resume from last.pt with --resume.")
        raise SystemExit(130) from None

    plot_history(pd.read_csv(history_path), output_dir / "training_curve.png")
    log(f"Training curve: {output_dir / 'training_curve.png'}")

    best_path = output_dir / "best.pt"
    if not best_path.exists():
        raise RuntimeError("Training finished without a checkpoint.")

    report: dict = {"checkpoint": str(best_path), "history": str(history_path)}
    if config.final_eval:
        log("Evaluating the best checkpoint on the official CIFAR-10 test set.")
        report["test_eval"] = run_evaluation(
            best_path,
            output_dir / "test_eval",
            epsilon=config.epsilon,
            pgd_steps=config.final_pgd_steps,
            batch_size=config.batch_size,
            data_dir=config.data_dir,
            seed=config.seed,
            num_workers=config.num_workers,
            save_examples=True,
        )
        summary = report["test_eval"]["summary"]
        log(
            f"Test set: clean {summary['clean_accuracy']:.2%}, "
            f"FGSM {summary['fgsm_accuracy']:.2%}, "
            f"PGD {summary['pgd_accuracy']:.2%}."
        )
    return report
