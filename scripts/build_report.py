"""Turn experiment outputs into report figures, result tables, and REPORT.md.

Usage:
    python scripts/build_report.py --results results.zip
    python scripts/build_report.py --results outputs/cpu_preview --note "Preliminary CPU run."

``--results`` is either the ``results.zip`` downloaded from the Colab
notebook or a folder that contains one sub-folder per training run. A run
folder is recognised by ``test_eval/summary.csv``.

Every number in the report is read from those files. Nothing is typed in by
hand, so re-running the script after new experiments keeps the report honest.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from string import Template

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from attack2defense.config import CIFAR10_CLASSES, ROBUST_ACCURACY_TARGET  # noqa: E402
from attack2defense.plotting import plot_overall  # noqa: E402

MODEL_NAMES = {
    "resnet18": "ResNet-18",
    "small_cnn": "Small CNN",
    "resnet50": "ResNet-50",
    "vit_b_16": "ViT-B/16",
}
MODE_NAMES = {
    "clean": "no defense",
    "fgsm": "FGSM adversarial training",
    "pgd": "PGD adversarial training",
    "gaussian": "Gaussian noise training",
}
MODE_ORDER = {"clean": 0, "fgsm": 1, "pgd": 2, "gaussian": 3}
TARGET_PERCENT = ROBUST_ACCURACY_TARGET * 100


@dataclass
class Run:
    name: str
    folder: Path
    summary: dict
    per_class: pd.DataFrame
    sweep: pd.DataFrame
    history: pd.DataFrame | None
    examples: pd.DataFrame | None
    config: dict = field(default_factory=dict)

    @property
    def mode(self) -> str:
        return str(self.summary["mode"])

    @property
    def label(self) -> str:
        model = MODEL_NAMES.get(str(self.summary["model"]), str(self.summary["model"]))
        return f"{model}, {MODE_NAMES.get(self.mode, self.mode)}"

    @property
    def short_label(self) -> str:
        name = MODE_NAMES.get(self.mode, self.mode)
        return name[:1].upper() + name[1:]


# --------------------------------------------------------------------------- loading


def _resolve_results(path: Path, workdir: Path) -> Path:
    if path.is_dir():
        return path
    if path.suffix == ".zip":
        target = workdir / path.stem
        if target.exists():
            shutil.rmtree(target)
        with zipfile.ZipFile(path) as archive:
            archive.extractall(target)
        return target
    raise SystemExit(f"--results must be a folder or a .zip file, not {path}")


def _read_csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def load_runs(results: Path) -> list[Run]:
    runs: list[Run] = []
    for summary_path in sorted(results.glob("*/test_eval/summary.csv")):
        folder = summary_path.parent.parent
        evaluation = summary_path.parent
        summary = pd.read_csv(summary_path).iloc[0].to_dict()
        sweep = _read_csv(evaluation / "sweep.csv")
        if sweep is None:
            sweep = pd.DataFrame([summary])
        config_path = folder / "config.json"
        config = pd.read_json(config_path, typ="series").to_dict() if config_path.exists() else {}
        runs.append(
            Run(
                name=folder.name,
                folder=folder,
                summary=summary,
                per_class=pd.read_csv(evaluation / "per_class.csv"),
                sweep=sweep.sort_values("epsilon").reset_index(drop=True),
                history=_read_csv(folder / "history.csv"),
                examples=_read_csv(evaluation / "attack_examples.csv"),
                config=config,
            )
        )
    if not runs:
        raise SystemExit(f"No runs found under {results}. Expected <run>/test_eval/summary.csv.")
    return sorted(runs, key=lambda run: (MODE_ORDER.get(run.mode, 9), run.name))


# --------------------------------------------------------------------------- figures


def _style(axis) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(True, alpha=0.3)


def _target_line(axis) -> None:
    axis.axhline(TARGET_PERCENT, color="#333333", linestyle="--", linewidth=1, label="80% target")


def figure_overall(runs: list[Run], path: Path) -> None:
    frame = pd.DataFrame([{**run.summary, "label": run.short_label} for run in runs])
    plot_overall(frame, path, "Test accuracy before and after attack (epsilon 2/255)")


def figure_epsilon(runs: list[Run], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    for run in runs:
        epsilons = [0.0, *(run.sweep["epsilon"] * 255).tolist()]
        pgd = [run.summary["clean_accuracy"] * 100, *(run.sweep["pgd_accuracy"] * 100).tolist()]
        axis.plot(epsilons, pgd, marker="o", label=run.short_label)
    _target_line(axis)
    axis.set_xlabel("Attack budget epsilon (x/255)")
    axis.set_ylabel("Accuracy after 10-step PGD (%)")
    axis.set_title("Robustness as the attack gets stronger")
    axis.set_ylim(0, 105)
    _style(axis)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def figure_per_class(runs: list[Run], path: Path) -> None:
    frame = pd.DataFrame(
        {run.short_label: (run.per_class["pgd_accuracy"] * 100).tolist() for run in runs},
        index=runs[0].per_class["class"].tolist(),
    )
    axis = frame.plot(kind="bar", figsize=(12, 5), width=0.8)
    _target_line(axis)
    axis.set_ylabel("Accuracy after PGD (%)")
    axis.set_title("Accuracy after PGD for every CIFAR-10 class (epsilon 2/255)")
    axis.set_ylim(0, 105)
    axis.set_xticks(range(len(frame.index)))
    axis.set_xticklabels(frame.index, rotation=30, ha="right")
    _style(axis)
    axis.legend()
    figure = axis.get_figure()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def figure_training(runs: list[Run], path: Path) -> bool:
    with_history = [run for run in runs if run.history is not None and not run.history.empty]
    if not with_history:
        return False
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for run in with_history:
        axes[0].plot(run.history["epoch"], run.history["val_clean"] * 100, marker=".", label=run.short_label)
        axes[1].plot(run.history["epoch"], run.history["val_pgd"] * 100, marker=".", label=run.short_label)
    titles = ("Validation accuracy, clean images", "Validation accuracy after PGD")
    for axis, title in zip(axes, titles, strict=True):
        _target_line(axis)
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylim(0, 105)
        _style(axis)
    axes[0].set_ylabel("Accuracy (%)")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return True


def build_figures(runs: list[Run], results: Path, figures: Path) -> dict[str, str]:
    if figures.exists():
        shutil.rmtree(figures)
    figures.mkdir(parents=True)
    names: dict[str, str] = {}
    main_runs = empirical_cifar_runs(runs)

    figure_overall(main_runs, figures / "overall_accuracy.png")
    figure_epsilon(main_runs, figures / "accuracy_vs_epsilon.png")
    figure_per_class(main_runs, figures / "per_class_pgd.png")
    if figure_training(main_runs, figures / "training_curves.png"):
        names["training"] = "training_curves.png"

    for run in runs:
        source = run.folder / "test_eval" / "attack_examples.png"
        if source.exists():
            shutil.copy(source, figures / f"attack_examples_{run.name}.png")
        per_class = run.folder / "test_eval" / "per_class.png"
        if per_class.exists():
            shutil.copy(per_class, figures / f"per_class_{run.name}.png")

    zeroshot = results / "zeroshot" / "zeroshot_comparison.png"
    if zeroshot.exists():
        shutil.copy(zeroshot, figures / "zeroshot_comparison.png")
        names["zeroshot"] = "zeroshot_comparison.png"

    certified = next(results.rglob("certified_accuracy.png"), None)
    if certified is not None:
        shutil.copy(certified, figures / "certified_accuracy.png")
        names["certified"] = "certified_accuracy.png"
    return names


# --------------------------------------------------------------------------- tables


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _markdown(frame: pd.DataFrame) -> str:
    header = "| " + " | ".join(str(column) for column in frame.columns) + " |"
    divider = "| " + " | ".join("---" for _ in frame.columns) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, divider, *rows])


def table_headline(runs: list[Run]) -> str:
    rows = []
    for run in runs:
        s = run.summary
        rows.append(
            {
                "Model": run.label,
                "Clean": _pct(s["clean_accuracy"]),
                "FGSM": _pct(s["fgsm_accuracy"]),
                "PGD-10": _pct(s["pgd_accuracy"]),
                "Drop under PGD": f"{(s['clean_accuracy'] - s['pgd_accuracy']) * 100:.1f} pts",
                "80% target": "met" if s["pgd_accuracy"] >= ROBUST_ACCURACY_TARGET else "not met",
            }
        )
    return _markdown(pd.DataFrame(rows))


def table_sweep(runs: list[Run]) -> str:
    epsilons = sorted({round(e * 255, 3) for run in runs for e in run.sweep["epsilon"]})
    rows = []
    for run in runs:
        lookup = {round(e * 255, 3): run.sweep.iloc[i] for i, e in enumerate(run.sweep["epsilon"])}
        row = {"Model": run.short_label, "Clean": _pct(run.summary["clean_accuracy"])}
        for epsilon in epsilons:
            entry = lookup.get(epsilon)
            label = f"{epsilon:g}/255"
            row[f"FGSM {label}"] = _pct(entry["fgsm_accuracy"]) if entry is not None else "-"
            row[f"PGD {label}"] = _pct(entry["pgd_accuracy"]) if entry is not None else "-"
        rows.append(row)
    return _markdown(pd.DataFrame(rows))


def table_per_class(runs: list[Run]) -> str:
    frame = pd.DataFrame({"Class": runs[0].per_class["class"]})
    for run in runs:
        frame[f"{run.short_label}: clean"] = run.per_class["clean_accuracy"].map(_pct)
        frame[f"{run.short_label}: PGD"] = run.per_class["pgd_accuracy"].map(_pct)
    return _markdown(frame)


def table_examples(runs: list[Run]) -> str:
    rows = []
    for run in runs:
        if run.examples is None:
            continue
        e = run.examples

        def correct(column: str, e: pd.DataFrame = e) -> str:
            return f"{int((e[column] == e['true_class']).sum())} / {len(e)}"

        rows.append(
            {
                "Model": run.short_label,
                "Correct before attack": correct("clean_prediction"),
                "Correct after FGSM": correct("fgsm_prediction"),
                "Correct after PGD": correct("pgd_prediction"),
            }
        )
    return _markdown(pd.DataFrame(rows)) if rows else "_No example grids in this results set._"


def table_training(runs: list[Run]) -> str:
    rows = []
    for run in runs:
        if run.history is None or run.history.empty:
            continue
        history = run.history
        score = history["val_clean"] if run.mode == "clean" else history["val_pgd"]
        best = history.loc[score.idxmax()]
        rows.append(
            {
                "Model": run.short_label,
                "Epochs": int(history["epoch"].max()),
                "Best epoch": int(best["epoch"]),
                "Training time": f"{history['seconds'].sum() / 60:.0f} min",
                "Optimizer": run.config.get("optimizer_name", "-"),
                "Schedule": run.config.get("scheduler_name", "-"),
            }
        )
    return _markdown(pd.DataFrame(rows)) if rows else "_No training history in this results set._"


def table_zeroshot(results: Path) -> str:
    frame = _read_csv(results / "zeroshot" / "zeroshot_summary.csv")
    if frame is None:
        return "_The zero-shot benchmark was not part of this results set._"
    rows = [
        {
            "Model": MODEL_NAMES.get(str(row["model"]), str(row["model"])),
            "Cat images": int(row["images"]),
            "Clean": _pct(row["clean_accuracy"]),
            "FGSM": _pct(row["fgsm_accuracy"]),
            "PGD": _pct(row["pgd_accuracy"]),
        }
        for _, row in frame.iterrows()
    ]
    return _markdown(pd.DataFrame(rows))


def _dataset(run: Run) -> str:
    return str(run.config.get("dataset", "cifar10"))


def empirical_cifar_runs(runs: list[Run]) -> list[Run]:
    """Main report tables: CIFAR-10 empirical attacks, not Gaussian or Imagenette."""

    selected = [run for run in runs if _dataset(run) == "cifar10" and run.mode != "gaussian"]
    return selected or runs


def table_certified(results: Path) -> str:
    paths = sorted(results.rglob("certify.csv"))
    if not paths:
        return ""
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame.insert(0, "Run", path.parent.parent.name)
        renamed = {"radius": "L2 radius", "certified_accuracy": "Certified accuracy"}
        frames.append(frame.rename(columns=renamed))
    combined = pd.concat(frames, ignore_index=True)
    combined["Certified accuracy"] = combined["Certified accuracy"].map(_pct)
    keep = [
        column
        for column in ("Run", "L2 radius", "Certified accuracy", "images")
        if column in combined.columns
    ]
    return _markdown(combined[keep])


def table_extra_attacks(results: Path) -> str:
    paths = sorted(results.rglob("extra_attacks.csv"))
    if not paths:
        return ""
    frames = [pd.read_csv(path) for path in paths]
    combined = pd.concat(frames, ignore_index=True)
    if "accuracy" in combined.columns:
        combined["accuracy"] = combined["accuracy"].map(_pct)
    return _markdown(combined)


def table_imagenette(runs: list[Run]) -> str:
    imagenette = [run for run in runs if _dataset(run) == "imagenette"]
    return table_headline(imagenette) if imagenette else ""


def section_or_skip(title: str, body: str, figure: str = "") -> str:
    if not body:
        return ""
    parts = [f"## {title}", body]
    if figure:
        parts.append(figure)
    return "\n\n".join(parts)


# --------------------------------------------------------------------------- narrative


def _verdict(runs: list[Run]) -> dict[str, str]:
    baseline = next((run for run in runs if run.mode == "clean"), None)
    defended = [run for run in runs if run.mode != "clean"]
    best = max(defended or runs, key=lambda run: run.summary["pgd_accuracy"])
    best_pgd = best.summary["pgd_accuracy"]

    if best_pgd >= ROBUST_ACCURACY_TARGET:
        verdict = (
            f"**Yes.** {best.label} keeps **{_pct(best_pgd)}** of the 10,000 test images "
            f"correct under a 10-step PGD attack at epsilon 2/255, above the 80% target."
        )
    else:
        verdict = (
            f"**Not yet.** The best model, {best.label}, keeps **{_pct(best_pgd)}** correct "
            f"under a 10-step PGD attack at epsilon 2/255. That is "
            f"{(ROBUST_ACCURACY_TARGET - best_pgd) * 100:.1f} points short of the 80% target."
        )

    if baseline is not None and best is not baseline:
        gain = (best_pgd - baseline.summary["pgd_accuracy"]) * 100
        cost = (baseline.summary["clean_accuracy"] - best.summary["clean_accuracy"]) * 100
        comparison = (
            f"Without a defense, the same architecture drops from "
            f"{_pct(baseline.summary['clean_accuracy'])} on clean images to "
            f"{_pct(baseline.summary['pgd_accuracy'])} under PGD. Adversarial training raises "
            f"accuracy under attack by **{gain:.1f} points**, at a cost of {cost:.1f} points "
            f"on clean images."
        )
    else:
        comparison = "No undefended baseline was included in this results set."

    return {
        "verdict": verdict,
        "comparison": comparison,
        "best_label": best.label,
        "best_name": best.name,
        "baseline_name": baseline.name if baseline else best.name,
    }


def _examples_observation(runs: list[Run]) -> str:
    sentences = []
    for run in runs:
        if run.examples is None:
            continue
        e = run.examples
        before = int((e["clean_prediction"] == e["true_class"]).sum())
        after = int((e["pgd_prediction"] == e["true_class"]).sum())
        sentences.append(
            f"{run.short_label}: {before} of {len(e)} correct before the attack, {after} after PGD."
        )
    return " ".join(sentences)


def _sweep_observation(runs: list[Run]) -> str:
    sentences = []
    for run in runs:
        strongest = run.sweep.iloc[-1]
        sentences.append(
            f"{run.short_label} goes from {_pct(run.summary['clean_accuracy'])} clean to "
            f"{_pct(strongest['pgd_accuracy'])} under PGD at {strongest['epsilon'] * 255:g}/255."
        )
    return "As epsilon grows, accuracy falls for every model. " + " ".join(sentences)


def _recipe(runs: list[Run]) -> str:
    lines = []
    for run in runs:
        c = run.config
        if not c:
            continue
        defense = {
            "clean": "clean images",
            "fgsm": f"FGSM-RS images at epsilon {float(c['epsilon']) * 255:g}/255",
            "pgd": f"{c.get('pgd_steps')}-step PGD images at epsilon {float(c['epsilon']) * 255:g}/255",
        }.get(run.mode, run.mode)
        lines.append(
            f"- **{run.short_label}**: {MODEL_NAMES.get(c['model'], c['model'])}, trained on {defense}; "
            f"{c['optimizer_name']} at learning rate {c['lr']:g}, weight decay {c['weight_decay']:g}, "
            f"{c.get('scheduler_name', 'cosine')} schedule, batch {c['batch_size']}, "
            f"{c['epochs']} epochs."
        )
    lines.append("- Every run uses random crops with 4-pixel padding and horizontal flips.")
    return "Training recipe:\n\n" + "\n".join(lines)


def _split(runs: list[Run]) -> str:
    per_class = int(runs[0].config.get("val_per_class", 100))
    validation = per_class * len(CIFAR10_CLASSES)
    return (
        f"Training uses {50_000 - validation:,} CIFAR-10 training images. The other "
        f"{validation:,} ({per_class} per class) are a validation split used only to choose "
        f"the best checkpoint."
    )


def _image(path: str, alt: str) -> str:
    return f"![{alt}](docs/figures/{path})"


def _example_section(runs: list[Run], figures: Path) -> str:
    blocks = []
    for run in runs:
        name = f"attack_examples_{run.name}.png"
        if (figures / name).exists():
            blocks.append(f"**{run.label}**\n\n{_image(name, 'Attack examples for ' + run.label)}")
    return "\n\n".join(blocks) if blocks else "_No example grids in this results set._"


def write_report(
    runs: list[Run],
    results: Path,
    figures: Path,
    extra_figures: dict[str, str],
    template_path: Path,
    output: Path,
    note: str,
) -> dict[str, str]:
    main_runs = empirical_cifar_runs(runs)
    imagenette = table_imagenette(runs)
    certified = table_certified(results)
    extra = table_extra_attacks(results)
    verdict = _verdict(main_runs)
    models = sorted(
        {MODEL_NAMES.get(str(run.summary["model"]), str(run.summary["model"])) for run in main_runs}
    )
    values = {
        "note": f"> {note}\n" if note else "",
        "models": ", ".join(models),
        "run_count": str(len(main_runs)),
        "test_images": f"{int(main_runs[0].summary['images']):,}",
        "verdict": verdict["verdict"],
        "comparison": verdict["comparison"],
        "headline_table": table_headline(main_runs),
        "sweep_table": table_sweep(main_runs),
        "per_class_table": table_per_class(main_runs),
        "examples_table": table_examples(main_runs),
        "training_table": table_training(main_runs),
        "zeroshot_table": table_zeroshot(results),
        "examples_section": _example_section(main_runs, figures),
        "examples_observation": _examples_observation(main_runs),
        "sweep_observation": _sweep_observation(main_runs),
        "recipe": _recipe(main_runs),
        "split": _split(main_runs),
        "certified_section": section_or_skip(
            "Certified defense (randomized smoothing)",
            certified,
            _image(extra_figures["certified"], "Certified accuracy versus L2 radius")
            if "certified" in extra_figures
            else "",
        ),
        "imagenette_section": section_or_skip("Imagenette generalization", imagenette),
        "extra_attacks_section": section_or_skip("CW and AutoAttack", extra),
        "fig_overall": _image("overall_accuracy.png", "Clean, FGSM and PGD accuracy for every model"),
        "fig_epsilon": _image("accuracy_vs_epsilon.png", "Accuracy after PGD against epsilon"),
        "fig_per_class": _image("per_class_pgd.png", "Accuracy after PGD for every class"),
        "fig_training": _image(extra_figures["training"], "Validation accuracy per epoch")
        if "training" in extra_figures
        else "_No training history in this results set._",
        "fig_zeroshot": _image(extra_figures["zeroshot"], "ImageNet ResNet-50 vs ViT-B/16 on CIFAR cats")
        if "zeroshot" in extra_figures
        else "",
        "classes": ", ".join(CIFAR10_CLASSES),
    }
    template = Template(template_path.read_text(encoding="utf-8"))
    output.write_text(template.substitute(values), encoding="utf-8")

    tables = "\n\n".join(
        part
        for part in (
            "# Result tables\n\nGenerated by `scripts/build_report.py`.",
            "## Test set, epsilon 2/255\n\n" + values["headline_table"],
            "## Attack strength sweep\n\n" + values["sweep_table"],
            "## Per class, epsilon 2/255\n\n" + values["per_class_table"],
            "## Example images\n\n" + values["examples_table"],
            "## Training\n\n" + values["training_table"],
            "## Zero-shot benchmark\n\n" + values["zeroshot_table"],
            ("## Certified defense\n\n" + certified) if certified else "",
            ("## Imagenette\n\n" + imagenette) if imagenette else "",
            ("## Extra attacks\n\n" + extra) if extra else "",
        )
        if part
    )
    (figures.parent / "results_tables.md").write_text(tables + "\n", encoding="utf-8")
    return values


README_START = "<!-- results:start -->"
README_END = "<!-- results:end -->"


def update_readme(readme: Path, values: dict[str, str], note: str) -> None:
    """Replace the generated results block of the README, leaving the rest untouched."""

    if not readme.exists():
        return
    text = readme.read_text(encoding="utf-8")
    if README_START not in text or README_END not in text:
        return
    block = "\n\n".join(
        part
        for part in (
            "## Results",
            f"> {note}" if note else "",
            values["verdict"],
            values["headline_table"],
            values["fig_overall"],
            "Before/after attack images, per-class results, and the epsilon sweep are in "
            "[REPORT.md](REPORT.md).",
        )
        if part
    )
    head, rest = text.split(README_START, 1)
    _, tail = rest.split(README_END, 1)
    readme.write_text(f"{head}{README_START}\n{block}\n{README_END}{tail}", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", required=True, help="results.zip from Colab, or a folder of runs.")
    parser.add_argument("--figures", default=str(ROOT / "docs" / "figures"))
    parser.add_argument("--template", default=str(ROOT / "docs" / "report_template.md"))
    parser.add_argument("--report", default=str(ROOT / "REPORT.md"))
    parser.add_argument("--readme", default=str(ROOT / "README.md"))
    parser.add_argument("--note", default="", help="Optional banner shown at the top of the report.")
    args = parser.parse_args()

    results = _resolve_results(Path(args.results), ROOT / "results")
    runs = load_runs(results)
    figures = Path(args.figures)
    extra = build_figures(runs, results, figures)
    values = write_report(runs, results, figures, extra, Path(args.template), Path(args.report), args.note)
    update_readme(Path(args.readme), values, args.note)
    print(f"Runs: {', '.join(run.name for run in runs)}")
    figure_count = len(list(figures.iterdir()))
    print(f"Wrote {args.report}, {figures.parent / 'results_tables.md'}, and {figure_count} figures.")


if __name__ == "__main__":
    main()
