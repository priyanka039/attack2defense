# Changelog

All notable changes to this project. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- **Light and dark mode.** A Dark/Light switch at the bottom of the sidebar (a sun/moon button on phones). The first visit follows the operating system setting; the choice is remembered in `localStorage["a2d.theme"]`. Charts switch palettes too, and the light accents are darkened to keep text contrast at 4.5:1 or better.
- **Start here guide** (`/guide`): the project's result in one chart, the four stages with links to their screens, three takeaways, a plain-English glossary and a setup checklist. First-time visitors land here.
- **Setup check.** `python -m attack2defense doctor` and `GET /api/setup` report whether CIFAR-10, the published results and each model's weights are installed, with the exact path for anything missing. The UI shows it as a status dot in the sidebar and a checklist on the guide.
- **`scripts/install_checkpoints.py`**: installs downloaded `best.pt` files (from a folder or a zip) into `checkpoints/<run>/best.pt`, validates each one, and adds its `config.json` from `results/`.
- **Attack lab compare mode**: attack two models with the same image and settings and see the results side by side.
- **Attack lab explanation**: a verdict banner and a "What just happened" card that describes the result in words, including the change in the model's confidence.
- **Presets** on Training (Quick demo, Stronger demo), Evaluation (Quick check, Thorough) and Certified defense (Quick, Careful), with a time estimate for this server's device.
- **InfoTip** definitions next to every technical term, usable with a keyboard and screen readers.
- Documentation: [ARCHITECTURE.md](docs/ARCHITECTURE.md), [API.md](docs/API.md), [USER_GUIDE.md](docs/USER_GUIDE.md), [DEVELOPMENT.md](docs/DEVELOPMENT.md), and screenshots of every screen.
- Tests for the setup endpoint, the guide's first-visit redirect, the InfoTip and the radio-group controls.

### Changed

- **Full UI redesign** with a dark "security lab" theme. Colour now carries meaning everywhere: teal for clean images and defenses, pink for attacks and fooled predictions, amber for the 80% target and warnings.
- Navigation is grouped into Learn, Experiment and Data, with a one-line hint per screen, a "login" chip on screens that need an account, and a mobile menu drawer.
- The Overview is now the **Dashboard**: headline numbers, before-and-after examples, a results table with target status, and the attack-strength sweep for the featured architecture.
- The Attack lab is a three-step flow (model, image, attack) with strength presets in brightness levels out of 255; advanced PGD settings are tucked away.
- Training, Evaluation, Certified defense, Models and data, Login and Not found were rebuilt with the new components, clearer empty states that say what to do next, and friendlier labels.
- Charts use the dark theme and draw the target as a labelled amber line.
- [checkpoints.md](docs/checkpoints.md) rewritten around the doctor command and the install script.

### Fixed

- FGSM requests from the UI always send one step.
- The Dashboard's attack-strength chart only plots the featured architecture, so the Small CNN demo runs no longer crowd the ResNet-18 curves.

## [1.0.0]

### Added

- `attack2defense` package: pixel-space FGSM and PGD, clean, FGSM, PGD and Gaussian-noise training, evaluation sweeps, randomized smoothing, Imagenette support, CW and AutoAttack benchmarks, and a CLI.
- FastAPI backend with accounts, background jobs for training, evaluation and certification, and CSV and JSON export.
- React workbench with Overview, Attack lab, Training, Evaluation, Certified defense, and Models and data.
- Colab notebook, report builder, `REPORT.md`, figures, evaluation guide and PRD checklist.
- GitHub Actions CI for linting, Python tests, frontend tests and the production build.

### Results

- ResNet-18 on the full CIFAR-10 test set at epsilon = 2/255: 93.6% clean and 6.1% under PGD-10 without a defense; 90.7% clean and 78.9% under PGD-10 with PGD adversarial training, 1.1 points short of the 80% target.
