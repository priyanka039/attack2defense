"""The work done by training and evaluation jobs."""

from __future__ import annotations

from pathlib import Path

from attack2defense.config import CIFAR10_CLASSES
from attack2defense.evaluate import evaluate_loader, evaluate_sweep, write_evaluation_files
from attack2defense.registry import MODE_LABELS
from attack2defense.train import TrainConfig, TrainingCancelled, TrainingHooks, run_training
from backend.app.schemas import CertifyRequest, EvaluationRequest, TrainingRequest
from backend.app.services.jobs import JobContext

_EVAL_BATCH = 100
_QUICK_TEST_PER_CLASS = 20


def training_run_dir(services, user_id: int, job_id: str) -> Path:
    return services.models.user_dir(user_id) / f"job_{job_id}"


def run_training_job(services, user_id: int, job_id: str, request: TrainingRequest, ctx: JobContext) -> dict:
    output_dir = training_run_dir(services, user_id, job_id)
    config = TrainConfig(
        model=request.model,
        mode=request.mode,
        epochs=request.epochs,
        batch_size=request.batch_size,
        lr=request.lr,
        optimizer_name=request.optimizer,
        epsilon=request.epsilon,
        pgd_steps=request.pgd_steps,
        adv_ratio=request.adv_ratio,
        sigma=request.sigma,
        val_pgd_steps=3,
        val_per_class=request.val_per_class,
        max_train_batches=request.max_train_batches,
        final_eval=False,
        output_dir=str(output_dir),
        data_dir=str(services.settings.data_dir),
        num_workers=0,
    )
    epochs = request.epochs

    def on_batch(epoch: int, batch: int, total: int) -> None:
        # Training is 90% of the job; the quick test evaluation is the rest.
        fraction = 0.9 * ((epoch - 1) + batch / total) / epochs
        ctx.progress(fraction, f"Epoch {epoch}/{epochs}, batch {batch}/{total}")

    def on_epoch(row: dict) -> None:
        ctx.add_history(row)
        ctx.progress(0.9 * row["epoch"] / epochs, f"Finished epoch {row['epoch']}/{epochs}", force=True)

    hooks = TrainingHooks(on_batch=on_batch, on_epoch=on_epoch, should_stop=ctx.cancelled)
    report = run_training(config, hooks)

    ctx.progress(0.92, "Testing the best checkpoint on held-out images", force=True)
    services.models.invalidate()
    entry = services.models.get(f"job_{job_id}", user_id)
    model = services.models.load(entry)
    loader = services.datasets.balanced_loader("cifar10", _QUICK_TEST_PER_CLASS, _EVAL_BATCH)
    result = evaluate_loader(
        model, loader, epsilon=request.epsilon, pgd_steps=10, device=services.models.device, desc="Test"
    )
    summary = write_evaluation_files(
        result,
        output_dir / "test_eval",
        model_name=request.model,
        mode=request.mode,
        class_names=list(CIFAR10_CLASSES),
    )
    return {"model_id": entry.id, "checkpoint": report["checkpoint"], "summary": summary}


def run_evaluation_job(services, user_id: int, request: EvaluationRequest, ctx: JobContext) -> dict:
    entries = [services.models.get(model_id, user_id) for model_id in request.model_ids]
    models = [services.models.load(entry) for entry in entries]
    loader = services.datasets.balanced_loader(request.dataset, request.images_per_class, _EVAL_BATCH)
    class_names = services.datasets.info(request.dataset)["classes"]

    results = []
    total = len(entries) * len(request.epsilons)
    for model_index, (entry, model) in enumerate(zip(entries, models, strict=True)):
        if ctx.cancelled():
            raise TrainingCancelled

        def report(done: int, _count: int, offset: int = model_index * len(request.epsilons)) -> None:
            ctx.progress(
                (offset + done) / total, f"Evaluated {offset + done} of {total} settings", force=True
            )

        ctx.progress(model_index * len(request.epsilons) / total, f"Attacking {entry.id}", force=True)
        rows = evaluate_sweep(
            model,
            loader,
            request.epsilons,
            request.pgd_steps,
            services.models.device,
            threshold=request.threshold,
            class_names=class_names,
            on_progress=report,
        )
        results.append(
            {
                "model_id": entry.id,
                "label": MODE_LABELS.get(entry.mode, entry.mode),
                "mode": entry.mode,
                "tag": entry.tag,
                "rows": rows,
            }
        )

    flags = [
        {"model_id": result["model_id"], "epsilon": row["epsilon"], "attack": attack}
        for result in results
        for row in result["rows"]
        for attack in ("fgsm", "pgd")
        if row[f"{attack}_below_threshold"]
    ]
    return {
        "dataset": request.dataset,
        "images": request.images_per_class * len(class_names),
        "threshold": request.threshold,
        "epsilons": request.epsilons,
        "pgd_steps": request.pgd_steps,
        "models": results,
        "flags": flags,
    }


def run_certify_job(services, user_id: int, request: CertifyRequest, ctx: JobContext) -> dict:
    from attack2defense.smoothing import certify_loader

    entry = services.models.get(request.model_id, user_id)
    model = services.models.load(entry)
    loader = services.datasets.balanced_loader(request.dataset, request.images_per_class, batch_size=1)

    def report(done: int, total: int | None) -> None:
        size = total or max(done, 1)
        ctx.progress(done / size, f"Certified {done} of {size} images", force=True)

    ctx.progress(0.0, f"Certifying {entry.id}", force=True)
    result = certify_loader(
        model,
        loader,
        sigma=request.sigma,
        n0=request.n0,
        n=request.n,
        alpha=request.alpha,
        batch_size=32,
        on_progress=report,
    )
    result.pop("rows", None)
    result["model_id"] = entry.id
    result["label"] = entry.to_dict()["label"]
    result["dataset"] = request.dataset
    result["kind"] = "certified"
    result["note"] = "Certified accuracy is a provable L2 radius, not empirical accuracy after FGSM or PGD."
    return result
