# API reference

Base URL: `http://localhost:8000/api`. Interactive docs, generated from the same code, are at [http://localhost:8000/docs](http://localhost:8000/docs).

All bodies are JSON. Epsilon and step sizes are fractions of the pixel range `[0, 1]`, so 2/255 is sent as `0.00784`.

## Conventions

### Authentication

Register or log in to get a token, then send it on every request:

```
Authorization: Bearer <access_token>
```

Tokens are HS256 JWTs and last 12 hours by default (`A2D_TOKEN_MINUTES`). Endpoints that are **public** work without a token; endpoints marked **login** return `401 not_authenticated` without one.

### Errors

Every failure has the same shape:

```json
{ "error": { "code": "too_many_epochs", "message": "Use at most 30 epochs here.", "field": "epochs" } }
```

`message` is written to be shown to a person as-is. `field` names the input at fault, or is `null`.

| Status | Typical `code` | Meaning |
| --- | --- | --- |
| 401 | `not_authenticated`, `wrong_credentials` | Missing, invalid or expired token; failed login |
| 404 | `model_not_found`, `training_job_not_found` | Unknown id, or a job that belongs to someone else |
| 409 | `model_has_no_weights`, `evaluation_not_finished`, `email_taken` | The request is valid but the server state does not allow it |
| 422 | `invalid_input`, `batch_too_large`, `too_many_batches` | Validation failed or a server limit was exceeded |
| 500 | `server_error` | Unexpected; details are in the backend log |

### Jobs

Training, evaluation and certification are long-running. Starting one returns `202` with a job, and the client polls it:

```json
{
  "id": "f6b9f5c4a9684a25b81c8b70e09c9930",
  "kind": "training",
  "status": "running",
  "params": { "model": "small_cnn", "mode": "pgd", "epochs": 2 },
  "progress": 0.78,
  "message": "Epoch 2/2, batch 29/40",
  "history": [{ "epoch": 1, "train_loss": 2.03, "val_clean": 0.28, "val_fgsm": 0.24, "val_pgd": 0.24 }],
  "result": null,
  "error": null,
  "created_at": "2026-11-12T10:32:04",
  "started_at": "2026-11-12T10:32:05",
  "finished_at": null
}
```

`status` moves through `queued` to `running` and ends as `succeeded`, `failed` or `cancelled`. While a cancel is in progress it is `cancelling`. `progress` is 0 to 1. One job runs at a time; others wait as `queued`.

Each job family has the same four routes: `POST` to start, `GET` to list your jobs (newest first), `GET /{id}` for one job, and `DELETE /{id}` to cancel.

---

## Meta

### `GET /health` (public)

```json
{ "status": "ok", "version": "1.0.0", "device": "cpu" }
```

### `GET /setup` (public)

Reports what is installed. The UI uses this for the setup status checklist.

```json
{
  "ready": false,
  "device": "cpu",
  "checks": [
    { "id": "dataset", "label": "CIFAR-10 test images", "ok": true, "hint": "" },
    { "id": "results", "label": "Published results", "ok": true, "hint": "" },
    { "id": "weights", "label": "Weights for live attacks", "ok": true, "hint": "" },
    { "id": "reported_weights", "label": "Reported ResNet-18 weights", "ok": false, "hint": "Download best.pt ..." }
  ],
  "runs": [{ "id": "resnet18_pgd", "architecture": "resnet18", "mode": "pgd", "has_weights": false, "has_results": true, "checkpoint": null }],
  "missing_reported_weights": [{ "id": "resnet18_pgd", "expected_path": "checkpoints/resnet18_pgd/best.pt" }]
}
```

The same report is available offline with `python -m attack2defense doctor`.

## Auth

### `POST /auth/register` (public), `201`

Body: `{ "email": "you@example.com", "password": "at least 8 characters" }`

Returns:

```json
{ "access_token": "eyJ...", "token_type": "bearer", "expires_in": 43200, "user": { "id": 1, "email": "you@example.com", "created_at": "..." } }
```

`409 email_taken` if the address is already registered. Reserved domains such as `.local` are rejected by email validation.

### `POST /auth/login` (public)

Same body and response as register. `401 wrong_credentials` on a wrong email or password; the message does not reveal which.

### `GET /auth/me` (login)

The current user.

## Results and models

### `GET /public/results` (public)

Everything the Dashboard needs:

```json
{
  "models": [ModelSummary, ...],
  "zeroshot": [{ "model": "resnet50", "clean_accuracy": 0.87, "...": "..." }],
  "certified": [{ "source": "resnet18_gaussian", "curve": [{ "radius": 0.0, "certified_accuracy": 0.71 }] }]
}
```

Only models that have a test summary are included.

### `GET /models` (public; includes your own runs when logged in)

A list of `ModelSummary`:

```json
{
  "id": "resnet18_pgd",
  "architecture": "resnet18",
  "dataset": "cifar10",
  "mode": "pgd",
  "label": "PGD adversarial training",
  "tag": "reported",
  "epsilon": 0.00784,
  "epochs": 30,
  "has_weights": false,
  "summary": { "model": "resnet18", "mode": "pgd", "epsilon": 0.00784, "pgd_steps": 10, "images": 10000,
               "clean_accuracy": 0.907, "fgsm_accuracy": 0.799, "pgd_accuracy": 0.789, "target": 0.8, "target_met": false },
  "sweep": [{ "epsilon": 0.00784, "clean_accuracy": 0.907, "fgsm_accuracy": 0.799, "pgd_accuracy": 0.789 }],
  "owner": "public"
}
```

`has_weights` decides whether a model can be attacked or evaluated. `owner` is `"you"` for your own training runs, whose ids look like `job_<id>`.

### `GET /models/{model_id}` (public)

`ModelSummary` plus `config` (model, mode, epsilon, pgd_steps, adv_ratio, epochs, batch_size, lr, optimizer_name, scheduler_name, seed), `history` (one row per epoch) and `has_examples`.

### `GET /models/{model_id}/examples.png` (public)

The before-and-after attack grid saved by the test evaluation. `404 examples_not_found` if there is none.

## Datasets

### `GET /datasets` (public)

The datasets this server knows about.

### `GET /datasets/{name}` (public)

```json
{ "name": "cifar10", "title": "CIFAR-10", "classes": ["airplane", "..."], "image_size": 32,
  "splits": { "train": 49000, "validation": 1000, "test": 10000 }, "description": "..." }
```

### `GET /datasets/{name}/samples` (public)

| Query | Default | Notes |
| --- | --- | --- |
| `label` | none | Filter by class name |
| `offset` | 0 | |
| `limit` | 24 | 1 to 64 |

```json
{ "total": 1000, "offset": 0, "items": [{ "index": 3, "label": "airplane", "image": "data:image/png;base64,..." }] }
```

`index` is the position in the test set; pass it to the attack endpoints as `sample_indices`.

## Attacks

### `POST /attacks/fgsm` and `POST /attacks/pgd` (login)

Runs synchronously and returns the adversarial images.

| Field | Type | Default | Rules |
| --- | --- | --- | --- |
| `model_id` | string | required | Must have weights |
| `dataset` | string | `cifar10` | |
| `epsilon` | number | 8/255 | 0 to 32/255 |
| `steps` | int | 10 | 1 to 100; FGSM always uses 1 |
| `alpha` | number | `2.5 * epsilon / steps` | PGD step size, above 0 and at most 32/255 |
| `random_start` | bool | true | PGD only |
| `sample_indices` | int[] | | Test-set indices, at most 16 |
| `image` | string | | Base64 PNG or JPEG data URL, used instead of `sample_indices` |
| `label` | string | | True class of an uploaded image, if known |

Give exactly one of `sample_indices` or `image`. An upload without a label is attacked away from whatever the model currently predicts.

Response:

```json
{
  "attack": "pgd",
  "model_id": "small_cnn_clean",
  "epsilon": 0.0314, "alpha": 0.00784, "steps": 10, "random_start": true,
  "seconds": 0.41,
  "success_per_step": [0.0, 0.0, 1.0, 1.0, "..."],
  "success_rate": 1.0,
  "items": [{
    "sample_index": 0,
    "true_label": "cat",
    "clean_image": "data:image/png;base64,...",
    "adversarial_image": "data:image/png;base64,...",
    "perturbation_image": "data:image/png;base64,...",
    "heatmap_image": "data:image/png;base64,...",
    "clean_prediction": { "label": "cat", "index": 3, "confidence": 0.71 },
    "adversarial_prediction": { "label": "automobile", "index": 1, "confidence": 0.44 },
    "clean_top": ["top 3 predictions"],
    "adversarial_top": ["top 3 predictions"],
    "fooled": true,
    "linf": 0.0314,
    "l2": 1.52
  }]
}
```

`success_per_step[k]` is the share of images misclassified after step `k + 1`. `perturbation_image` is the change amplified to full contrast; `heatmap_image` shows where it is largest.

## Training

### `POST /training/jobs` (login), `202`

| Field | Default | Rules |
| --- | --- | --- |
| `name` | none | At most 60 characters |
| `model` | `small_cnn` | `small_cnn` or `resnet18` |
| `mode` | `pgd` | `clean`, `fgsm`, `pgd`, `gaussian` |
| `epsilon` | 2/255 | 0 to 32/255 |
| `pgd_steps` | 3 | 1 to 20 |
| `adv_ratio` | 1.0 | 0 to 1: share of each batch replaced by attacked images |
| `sigma` | 0.25 | Noise level for `gaussian` |
| `epochs` | 2 | At most `A2D_MAX_TRAIN_EPOCHS` (30) |
| `batch_size` | 128 | 8 to 512 |
| `lr` | 0.001 | |
| `optimizer` | `adamw` | `adamw` or `sgd` |
| `max_train_batches` | 40 | Required; at most `A2D_MAX_TRAIN_BATCHES` (400) |
| `val_per_class` | 20 | 1 to 100 |

Progress messages read `Epoch e/E, batch b/B`. Training is 90% of the progress bar; a quick test on 200 held-out images is the rest. `history` grows by one row per epoch. On success:

```json
{ "model_id": "job_f6b9...", "checkpoint": ".../best.pt", "summary": { "clean_accuracy": 0.385, "fgsm_accuracy": 0.29, "pgd_accuracy": 0.29 } }
```

The new model immediately appears in `GET /models` for its owner.

Also: `GET /training/jobs`, `GET /training/jobs/{id}`, `DELETE /training/jobs/{id}`.

## Evaluation

### `POST /evaluations` (login), `202`

| Field | Default | Rules |
| --- | --- | --- |
| `model_ids` | required | 1 to 6 models, each with weights |
| `dataset` | `cifar10` | |
| `epsilons` | [2/255, 4/255, 8/255] | 1 to 8 values; sorted and de-duplicated |
| `pgd_steps` | 10 | 1 to 50 |
| `threshold` | 0.8 | Robust accuracy below this is flagged |
| `images_per_class` | 20 | At most `A2D_MAX_EVAL_IMAGES_PER_CLASS` (200) |

Every model is attacked on the same class-balanced images. Result:

```json
{
  "dataset": "cifar10", "images": 50, "threshold": 0.5, "epsilons": [0.0078, 0.0157, 0.0314], "pgd_steps": 10,
  "models": [{ "model_id": "small_cnn_clean", "label": "No defense", "mode": "clean", "tag": "...",
               "rows": [{ "epsilon": 0.0078, "clean_accuracy": 0.6, "fgsm_accuracy": 0.24, "pgd_accuracy": 0.18,
                          "fgsm_below_threshold": true, "pgd_below_threshold": true, "per_class": ["..."] }] }],
  "flags": [{ "model_id": "small_cnn_clean", "epsilon": 0.0078, "attack": "pgd" }]
}
```

### `GET /evaluations/{id}/export?format=csv|json` (login)

Downloads a finished evaluation. CSV has one row per model and epsilon. `409 evaluation_not_finished` if it is still running.

Also: `GET /evaluations`, `GET /evaluations/{id}`, `DELETE /evaluations/{id}`.

## Certification

### `POST /certify` (login), `202`

| Field | Default | Rules |
| --- | --- | --- |
| `model_id` | required | Must have weights |
| `sigma` | 0.25 | Noise level; match the sigma the model was trained with |
| `n0` | 25 | 10 to 200 samples to pick the class |
| `n` | 100 | 20 to 2000 samples to bound its probability |
| `alpha` | 0.001 | Failure probability of the bound |
| `images_per_class` | 5 | 1 to 50 |

Result:

```json
{
  "model_id": "small_cnn_clean", "label": "No defense", "dataset": "cifar10", "kind": "certified",
  "sigma": 0.25, "n0": 25, "n": 100, "alpha": 0.001, "images": 50, "naive_accuracy": 0.42,
  "curve": [{ "radius": 0.0, "certified_accuracy": 0.3, "images": 50 }, { "radius": 0.05, "certified_accuracy": 0.26, "images": 50 }],
  "note": "Certified accuracy is a provable L2 radius, not empirical accuracy after FGSM or PGD."
}
```

`curve` covers radii 0 to 1 in steps of 0.05. `naive_accuracy` is the share of images whose smoothed prediction was correct before any radius is required.

Also: `GET /certify`, `GET /certify/{id}`, `DELETE /certify/{id}`.

## Configuration

| Variable | Default | Effect |
| --- | --- | --- |
| `A2D_SECRET_KEY` | development key (warns) | JWT signing key; set this in production |
| `A2D_TOKEN_MINUTES` | 720 | Token lifetime |
| `A2D_DATABASE_URL` | `sqlite:///backend/attack2defense.db` | Any SQLAlchemy URL |
| `A2D_DATA_DIR` | `data` | Where CIFAR-10 is stored |
| `A2D_DOWNLOAD_DATA` | true | Download CIFAR-10 when missing |
| `A2D_CHECKPOINT_ROOTS` | `["checkpoints", "outputs/cpu_preview"]` | JSON list of folders scanned for `<run>/best.pt` |
| `A2D_RESULTS_ROOTS` | `["results"]` | JSON list of folders scanned for results |
| `A2D_RUNS_DIR` | `outputs/api_runs` | Where users' training runs go |
| `A2D_CORS_ORIGINS` | `["http://localhost:5173", "http://127.0.0.1:5173"]` | Allowed UI origins |
| `A2D_MAX_ATTACK_BATCH` | 16 | Images per attack request |
| `A2D_MAX_EVAL_IMAGES_PER_CLASS` | 200 | |
| `A2D_MAX_TRAIN_EPOCHS` | 30 | |
| `A2D_MAX_TRAIN_BATCHES` | 400 | Batches per epoch |
| `A2D_INLINE_JOBS` | false | Run jobs in the request thread (tests) |

Variables can also go in `backend/.env`.
