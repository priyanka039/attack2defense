# PRD acceptance checklist

Maps each Product Requirements Document item to where it is implemented.

## Must have

| Requirement | Where |
| --- | --- |
| FGSM, sign of gradient | [`attack2defense/attacks.py`](../attack2defense/attacks.py) `fgsm_attack`; API `POST /api/attacks/fgsm` |
| PGD, multi-step, L-inf projected | `pgd_attack` / `run_attack`; API `POST /api/attacks/pgd` |
| Baseline CIFAR-10 training | `python -m attack2defense train --mode clean`; API `POST /api/training/jobs` |
| Adversarial training | `--mode fgsm` or `--mode pgd`, with `--adv-ratio` |
| Evaluation sweep | `evaluate --sweep`; API `POST /api/evaluations` |
| FastAPI backend | [`backend/app/`](../backend/app/) |
| React + Tailwind frontend | [`frontend/`](../frontend/) |
| Auth and sessions | JWT in `POST /api/auth/login`; `RequireAuth` in the UI |
| Tests | `pytest tests` (pipeline + API); `npm test` in `frontend/` |
| README, architecture, reproduction | [README.md](../README.md) |

## User stories

| ID | Status | Notes |
| --- | --- | --- |
| US-01 Baseline and data pipeline | Met | CIFAR-10 loaders with a seeded train/val/test split. Imagenette is the ImageNet subset. Checkpoints stored and listed by the registry and `GET /api/models`. |
| US-02 FGSM | Met | Configurable epsilon, side-by-side images, perturbation heatmap, torchattacks numerical check in `tests/test_reference_attacks.py`. |
| US-03 PGD | Met | Epsilon, alpha, steps, random start. Per-step success and run time in `AttackResult` and the API. |
| US-04 Adversarial training | Met | Attack config + `adv_ratio`. Tagged checkpoints. History CSV and UI curves. |
| US-05 Evaluation | Met | Clean / FGSM / PGD at three or more epsilons, table + chart, CSV/JSON export, configurable threshold. |
| US-06 Full stack, auth, session | Met | Token login. Guests see public demo results. Run creation is per user. API errors use `{error: {code, message, field}}`. |
| US-07 Reproducibility | Met | README steps, architecture diagram, `pytest` and `npm test`. |
| US-08 Certified defense | Met as stretch | Randomized smoothing in `attack2defense/smoothing.py`, CLI `certify`, UI `/certified`. Additive; empirical flow does not depend on it. |

## Should have

| Item | Where |
| --- | --- |
| ImageNet subset | Imagenette via `--dataset imagenette` |
| Live epsilon / step sliders | Attack lab, 300 ms re-attack |
| CSV / JSON export | `GET /api/evaluations/{id}/export` |
| Training progress in the UI | Training screen polls jobs |
| Foolbox / torchattacks cross-check | torchattacks in `tests/test_reference_attacks.py` |

## Could have

| Item | Where |
| --- | --- |
| Certified defense | Smoothing + `/certified` |
| CW / AutoAttack | `python -m attack2defense benchmark-attacks` |
| Perturbation heatmap | Attack lab `heatmap_image` |
| Experiment registry | `attack2defense/registry.py` and Models screen |
